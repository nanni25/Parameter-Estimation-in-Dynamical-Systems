import numpy as np
import json
import roadrunner
from simulation import evaluate_loss
from visualization import plot_results, plot_timeline

# --- CONFIGURATION ---
MODEL_PATH = "Models/R-HSA-5653890_Modified.sbml"

SIMULATION_TIME = 100.0  
SIMULATION_STEPS = 100   
POPULATION_SIZE = 100      
NUM_GENERATIONS = 500      
LEARNING_RATE = 0.1       
SIGMA = 0.1

def main():
    np.random.seed(11)

    print("Loading target values from LLM...")
    with open("targets.json", "r") as f:
        targets = json.load(f)

    # Initialize Engine Once
    print("Compiling SBML in libRoadRunner...")
    rr = roadrunner.RoadRunner(MODEL_PATH)

    all_sbml_params = rr.model.getGlobalParameterIds()
    PARAMS_TO_OPTIMIZE = [
        p for p in all_sbml_params 
        if p.startswith(('lambda_', 'K_in_', 'K_out_'))
    ]
    
    MEAN_VARIABLES = [f"y_{sp.replace('species_', '')}" for sp in targets.keys()]

    rr.timeCourseSelections = MEAN_VARIABLES

    num_params = len(PARAMS_TO_OPTIMIZE)
    theta = np.zeros(num_params) 
    half_pop = POPULATION_SIZE // 2 
    
    m = np.zeros(num_params)
    v = np.zeros(num_params)
    beta1, beta2, adam_epsilon = 0.9, 0.999, 1e-8
    
    history_best_loss, history_mean_loss = [], []
    
    print(f"Starting for {NUM_GENERATIONS} gens. Pop Size: {POPULATION_SIZE}")
    
    for epoch in range(NUM_GENERATIONS):
        decay_factor = np.exp(-3.0 * (epoch / NUM_GENERATIONS)) 
        current_lr = LEARNING_RATE * decay_factor
        current_sigma = max(SIGMA * decay_factor, 0.001) 
        
        noise = np.random.randn(half_pop, num_params) 
        epsilons = np.concatenate([noise, -noise]) 
        
        raw_losses = np.zeros(POPULATION_SIZE)
        for i in range(POPULATION_SIZE):
            theta_try = theta + current_sigma * epsilons[i]
            # Pass everything to simulation module
            raw_losses[i] = evaluate_loss(
                rr, theta_try, targets, PARAMS_TO_OPTIMIZE, 
                MEAN_VARIABLES, SIMULATION_TIME, SIMULATION_STEPS
            )
            
        losses = np.zeros(POPULATION_SIZE)
        losses[np.argsort(raw_losses)] = np.linspace(0.5, -0.5, POPULATION_SIZE)
        
        step = np.dot(losses, epsilons) / (POPULATION_SIZE * current_sigma)
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
        
        if (epoch + 1) % 10 == 0:
            print(f"Gen {epoch+1:03d}/{NUM_GENERATIONS} | LR: {current_lr:.4f} | Sig: {current_sigma:.4f} | Best Loss: {min_loss:.6f}")
        
    print("\nOptimization Complete:")
    final_params = 10 ** theta
    
    print(f"{'Parameter':<25} | {'Optimized Value'}")
    print("-" * 45)
    for name, opt_val in zip(PARAMS_TO_OPTIMIZE, final_params):
        print(f"{name:<25} | {opt_val:.6f}")

    print("\nSimulating final parameters to compare against Targets...")
    rr.resetAll()
    for param_id, param_val in zip(PARAMS_TO_OPTIMIZE, final_params):
        rr.setValue(param_id, param_val)
        
    result = rr.simulate(0, SIMULATION_TIME, steps=SIMULATION_STEPS)
    simulated_means = np.array(result)[-1]

    target_values = [targets[var.replace("y_", "species_")] for var in MEAN_VARIABLES]

    print(f"\n{'Species (Target)':<18} | {'LLM Target':<12} | {'Simulated Mean'}")
    print("-" * 55)
    for var, target_val, sim_val in zip(MEAN_VARIABLES, target_values, simulated_means):
        print(f"{var:<18} | {target_val:<12.6f} | {sim_val:.6f}")

    # plot_results(history_best_loss, history_mean_loss, MEAN_VARIABLES, target_values, simulated_means)

    # plot_timeline(final_params, targets, MODEL_PATH, PARAMS_TO_OPTIMIZE, MEAN_VARIABLES, SIMULATION_TIME, SIMULATION_STEPS)

if __name__ == "__main__":
    main()