import numpy as np
import json
import roadrunner
import argparse
import copy
from simulation import evaluate_loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--targets_path", required=True)
    parser.add_argument("--sim_time", type=float, required=True)
    parser.add_argument("--sim_steps", type=int, required=True)
    parser.add_argument("--pop_size", type=int, required=True)
    parser.add_argument("--generations", type=int, required=True)
    parser.add_argument("--learning_rate", type=float, required=True)
    parser.add_argument("--sigma", type=float, required=True)    
    parser.add_argument("--patience", type=int, default=50, help="Generations to wait for improvement before stopping")
    parser.add_argument("--min_delta", type=float, default=1e-5, help="Minimum decrease in loss to count as improvement")
    
    args = parser.parse_args()

    np.random.seed(11)

    print("Loading target values from LLM...")
    with open(args.targets_path, "r") as f:
        targets = json.load(f)

    # Initialize Engine Once
    print("Compiling SBML in libRoadRunner...")
    rr = roadrunner.RoadRunner(args.model_path)

    all_sbml_params = rr.model.getGlobalParameterIds()
    PARAMS_TO_OPTIMIZE = [
        p for p in all_sbml_params 
        if p.startswith(('lambda_', 'K_in_', 'K_out_'))
    ]
    
    MEAN_VARIABLES = [f"y_{sp.replace('species_', '')}" for sp in targets.keys()]

    rr.timeCourseSelections = MEAN_VARIABLES

    num_params = len(PARAMS_TO_OPTIMIZE)
    theta = np.zeros(num_params) 
    half_pop = args.pop_size // 2 
    
    m = np.zeros(num_params)
    v = np.zeros(num_params)
    beta1, beta2, adam_epsilon = 0.9, 0.999, 1e-8
    
    history_best_loss, history_mean_loss = [], []
    
    best_global_loss = float('inf')
    best_theta = np.copy(theta)
    epochs_without_improvement = 0
    
    print(f"Starting for {args.generations} gens. Pop Size: {args.pop_size}")
    
    for epoch in range(args.generations):
        decay_factor = np.exp(-3.0 * (epoch / args.generations)) 
        current_lr = args.learning_rate * decay_factor
        current_sigma = max(args.sigma * decay_factor, 0.001) 
        
        noise = np.random.randn(half_pop, num_params) 
        epsilons = np.concatenate([noise, -noise]) 
        
        raw_losses = np.zeros(args.pop_size)
        for i in range(args.pop_size):
            theta_try = theta + current_sigma * epsilons[i]
            # Pass everything to simulation module
            raw_losses[i] = evaluate_loss(
                rr, theta_try, targets, PARAMS_TO_OPTIMIZE, 
                MEAN_VARIABLES, args.sim_time, args.sim_steps
            )
            
        losses = np.zeros(args.pop_size)
        losses[np.argsort(raw_losses)] = np.linspace(0.5, -0.5, args.pop_size)
        
        step = np.dot(losses, epsilons) / (args.pop_size * current_sigma)
        g = -step
        
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * (g ** 2)
        m_hat = m / (1 - beta1 ** (epoch + 1))
        v_hat = v / (1 - beta2 ** (epoch + 1))
        
        theta = theta - current_lr * m_hat / (np.sqrt(v_hat) + adam_epsilon)
        theta = np.clip(theta, -6.0, 6.0)
        
        min_loss, mean_loss = np.min(raw_losses), np.mean(raw_losses)
        history_best_loss.append(min_loss)
        history_mean_loss.append(mean_loss)
        
        if min_loss < (best_global_loss - args.min_delta):
            best_global_loss = min_loss
            best_theta = np.copy(theta) # Save the best parameters
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            
        if (epoch + 1) % 10 == 0:
            print(f"Gen {epoch+1:03d}/{args.generations} | LR: {current_lr:.4f} | Sig: {current_sigma:.4f} | Best Loss: {min_loss:.6f}")
            
        if epochs_without_improvement >= args.patience:
            print(f"\n[!] Early stopping triggered at Gen {epoch+1}!")
            print(f"[!] No improvement greater than {args.min_delta} for {args.patience} generations.")
            break
        
    print("\nOptimization Complete:")
    
    final_params = 10 ** best_theta 
    
    print(f"{'Parameter':<25} | {'Optimized Value'}")
    print("-" * 45)
    for name, opt_val in zip(PARAMS_TO_OPTIMIZE, final_params):
        print(f"{name:<25} | {opt_val:.6f}")

    print("\nSimulating final parameters to compare against Targets...")
    rr.resetAll()
    for param_id, param_val in zip(PARAMS_TO_OPTIMIZE, final_params):
        rr.setValue(param_id, param_val)
        
    result = rr.simulate(0, args.sim_time, steps=args.sim_steps)
    simulated_means = np.array(result)[-1]

    target_values = [targets[var.replace("y_", "species_")] for var in MEAN_VARIABLES]

    print(f"\n{'Species (Target)':<18} | {'LLM Target':<12} | {'Simulated Mean'}")
    print("-" * 55)
    for var, target_val, sim_val in zip(MEAN_VARIABLES, target_values, simulated_means):
        print(f"{var:<18} | {target_val:<12.6f} | {sim_val:.6f}")

if __name__ == "__main__":
    main()