import numpy as np
import config
import json
import roadrunner
from simulation import evaluate_loss
from visualization import plot_results, plot_timeline

def main():
    np.random.seed(11)

    print("Loading target values from LLM...")
    with open("targets.json", "r") as f:
        targets = json.load(f)
    
    num_params = len(config.PARAMS_TO_OPTIMIZE)
    theta = np.zeros(num_params) 
    
    pop_size = config.POPULATION_SIZE
    half_pop = pop_size // 2 
    
    m = np.zeros(num_params)
    v = np.zeros(num_params)
    beta1 = 0.9
    beta2 = 0.999
    adam_epsilon = 1e-8
    
    initial_lr = config.LEARNING_RATE
    initial_sigma = config.SIGMA
    
    history_best_loss = []
    history_mean_loss = []
    
    print(f"Starting for {config.NUM_GENERATIONS} gens. Pop Size: {pop_size}")
    
    for epoch in range(config.NUM_GENERATIONS):
        decay_factor = np.exp(-3.0 * (epoch / config.NUM_GENERATIONS)) 
        current_lr = initial_lr * decay_factor
        current_sigma = max(initial_sigma * decay_factor, 0.001) 
        
        noise = np.random.randn(half_pop, num_params) 
        epsilons = np.concatenate([noise, -noise]) 
        
        raw_losses = np.zeros(pop_size)
        for i in range(pop_size):
            theta_try = theta + current_sigma * epsilons[i]
            raw_losses[i] = evaluate_loss(theta_try, targets)
            
        losses = np.zeros(pop_size)
        losses[np.argsort(raw_losses)] = np.linspace(0.5, -0.5, pop_size)
        
        step = np.dot(losses, epsilons) / (pop_size * current_sigma)
        g = -step
        
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * (g ** 2)
        m_hat = m / (1 - beta1 ** (epoch + 1))
        v_hat = v / (1 - beta2 ** (epoch + 1))
        
        theta = theta - current_lr * m_hat / (np.sqrt(v_hat) + adam_epsilon)
        
        theta = np.clip(theta, -6.0, 6.0)
        
        min_loss = np.min(raw_losses)
        mean_loss = np.mean(raw_losses)
        
        history_best_loss.append(min_loss)
        history_mean_loss.append(mean_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Gen {epoch+1:03d}/{config.NUM_GENERATIONS} | LR: {current_lr:.4f} | Sig: {current_sigma:.4f} | Best Loss: {min_loss:.6f}")
        
    print("\nOptimization Complete:")
    final_params = 10 ** theta
    
    print(f"{'Parameter':<15} | {'Optimized Value'}")
    print("-" * 35)
    for name, opt_val in zip(config.PARAMS_TO_OPTIMIZE, final_params):
        print(f"{name:<15} | {opt_val:.6f}")

    print("\nSimulating final parameters to compare against Targets...")
    
    rr = roadrunner.RoadRunner(config.MODEL_PATH)
    rr.timeCourseSelections = config.MEAN_VARIABLES
    rr.resetAll()
    
    for param_id, param_val in zip(config.PARAMS_TO_OPTIMIZE, final_params):
        rr.setValue(param_id, param_val)
        
    result = rr.simulate(0, config.SIMULATION_TIME, steps=config.SIMULATION_STEPS)
    simulated_means = np.array(result)[-1]

    target_values = []
    for var in config.MEAN_VARIABLES:
        species_id = var.replace("y_", "species_")
        target_values.append(targets[species_id])

    print(f"\n{'Species (Target)':<18} | {'LLM Target':<12} | {'Simulated Mean'}")
    print("-" * 55)
    for var, target_val, sim_val in zip(config.MEAN_VARIABLES, target_values, simulated_means):
        print(f"{var:<18} | {target_val:<12.6f} | {sim_val:.6f}")

    # plot_results(history_best_loss, history_mean_loss, config.MEAN_VARIABLES, target_values, simulated_means)
    # plot_timeline(final_params, targets)

if __name__ == "__main__":
    main()