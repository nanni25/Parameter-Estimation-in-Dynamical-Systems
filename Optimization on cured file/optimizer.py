import numpy as np
import config
from simulation import evaluate_loss, generate_synthetic_targets

def main():
    np.random.seed(11)

    print("Generating pure synthetic targets (no noise)...")
    targets = generate_synthetic_targets(config.TRUE_PARAMS)
    print(f"Synthetic Targets Matrix Shape: {targets.shape}\n")

    num_params = len(config.PARAMS_TO_OPTIMIZE)
    theta = np.zeros(num_params) 
    
    pop_size = config.POPULATION_SIZE
    half_pop = pop_size // 2 
    
    print(f"Starting for {config.NUM_GENERATIONS} gens. Pop Size: {pop_size} | LR: {config.LEARNING_RATE} | Sigma: {config.SIGMA}\n")
    
    for epoch in range(config.NUM_GENERATIONS):
        
        noise = np.random.randn(half_pop, num_params) 
        epsilons = np.concatenate([noise, -noise]) 
        
        raw_losses = np.zeros(pop_size)
        for i in range(pop_size):
            theta_try = theta + config.SIGMA * epsilons[i]
            raw_losses[i] = evaluate_loss(theta_try)
            
        losses = np.zeros(pop_size)
        losses[np.argsort(raw_losses)] = np.linspace(0.5, -0.5, pop_size)
        
        step = np.dot(losses, epsilons) / (pop_size * config.SIGMA)
        theta = theta + config.LEARNING_RATE * step
        
        min_loss = np.min(raw_losses)
        mean_loss = np.mean(raw_losses)
        print(f"Generation {epoch+1:02d}/{config.NUM_GENERATIONS} | Best Loss: {min_loss:.4e} | Mean Loss: {mean_loss:.4e}")
        
    print("\nOptimization Complete:")
    final_params = np.exp(theta)
    
    print(f"{'Parameter':<15} | {'True Value':<12} | {'Optimized Value'}")
    print("-" * 50)
    for name, true_val, opt_val in zip(config.PARAMS_TO_OPTIMIZE, config.TRUE_PARAMS, final_params):
        print(f"{name:<15} | {true_val:<12.6f} | {opt_val:.6f}")

if __name__ == "__main__":
    main()