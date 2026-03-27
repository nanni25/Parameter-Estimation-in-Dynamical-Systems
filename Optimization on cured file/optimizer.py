import numpy as np
import config
from simulation import evaluate_loss

def main():
    np.random.seed(11)

    num_params = len(config.PARAMS_TO_OPTIMIZE)
    
    # I chose zeros because exp(0) = 1.0 for all parameters.
    theta = np.zeros(num_params) 
    
    pop_size = config.POPULATION_SIZE
    half_pop = pop_size // 2 
    
    print(f"Starting for {config.NUM_GENERATIONS} gens. Pop Size: {pop_size} | Learning Rate: {config.LEARNING_RATE} | Sigma: {config.SIGMA}\n")
    
    for epoch in range(config.NUM_GENERATIONS):
        
        # 1. Antithetic / Mirrored Sampling
        noise = np.random.randn(half_pop, num_params) 
        epsilons = np.concatenate([noise, -noise]) 
        
        # 2. Fitness Evaluation
        raw_losses = np.zeros(pop_size)
        for i in range(pop_size):
            theta_try = theta + config.SIGMA * epsilons[i]
            raw_losses[i] = evaluate_loss(theta_try)
            
        # 3. Fitness Shaping (Inlined)
        losses = np.zeros(pop_size)
        losses[np.argsort(raw_losses)] = np.linspace(0.5, -0.5, pop_size)
        
        # 4. Parameter Update
        step = np.dot(losses, epsilons) / (pop_size * config.SIGMA)
        theta = theta + config.LEARNING_RATE * step
        
        # Reporting
        min_loss = np.min(raw_losses)
        mean_loss = np.mean(raw_losses)
        print(f"Generation {epoch+1:02d}/{config.NUM_GENERATIONS} | Best Loss: {min_loss:.4e} | Mean Loss: {mean_loss:.4e}")
        
    print("\nOptimization Complete :")
    final_params = np.exp(theta)
    for name, val in zip(config.PARAMS_TO_OPTIMIZE, final_params):
        print(f"  {name}: {val:.6f}")

if __name__ == "__main__":
    main()