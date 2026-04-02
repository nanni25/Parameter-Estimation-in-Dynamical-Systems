import numpy as np
import config
from simulation import evaluate_loss, generate_synthetic_targets

def main():
    np.random.seed(11)

    print("Generating pure synthetic targets (no noise)...")
    # 1. Generate and store targets locally in this variable
    targets = generate_synthetic_targets(config.TRUE_PARAMS)
    
    num_params = len(config.PARAMS_TO_OPTIMIZE)
    theta = np.zeros(num_params) 
    
    pop_size = config.POPULATION_SIZE
    half_pop = pop_size // 2 
    
    # Adam Optimizer State Variables
    m = np.zeros(num_params)
    v = np.zeros(num_params)
    beta1 = 0.9
    beta2 = 0.999
    adam_epsilon = 1e-8
    
    # Base Hyperparameters
    initial_lr = config.LEARNING_RATE
    initial_sigma = config.SIGMA
    
    print(f"Starting for {config.NUM_GENERATIONS} gens. Pop Size: {pop_size}")
    
    for epoch in range(config.NUM_GENERATIONS):
        # 1. Decay Schedules: Reduce LR and Sigma as training progresses
        decay_factor = np.exp(-3.0 * (epoch / config.NUM_GENERATIONS)) 
        current_lr = initial_lr * decay_factor
        current_sigma = max(initial_sigma * decay_factor, 0.001) 
        
        noise = np.random.randn(half_pop, num_params) 
        epsilons = np.concatenate([noise, -noise]) 
        
        raw_losses = np.zeros(pop_size)
        for i in range(pop_size):
            theta_try = theta + current_sigma * epsilons[i]
            # 2. Pass the targets explicitly into the loss function!
            raw_losses[i] = evaluate_loss(theta_try, targets)
            
        # Fitness shaping
        losses = np.zeros(pop_size)
        losses[np.argsort(raw_losses)] = np.linspace(0.5, -0.5, pop_size)
        
        # Calculate step
        step = np.dot(losses, epsilons) / (pop_size * current_sigma)
        g = -step
        
        # Adam Update
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * (g ** 2)
        m_hat = m / (1 - beta1 ** (epoch + 1))
        v_hat = v / (1 - beta2 ** (epoch + 1))
        
        theta = theta - current_lr * m_hat / (np.sqrt(v_hat) + adam_epsilon)
        
        min_loss = np.min(raw_losses)
        
        if (epoch + 1) % 10 == 0:
            print(f"Gen {epoch+1:03d}/{config.NUM_GENERATIONS} | LR: {current_lr:.4f} | Sig: {current_sigma:.4f} | Best Loss: {min_loss:.6f}")
        
    print("\nOptimization Complete:")
    final_params = np.exp(theta)
    
    print(f"{'Parameter':<15} | {'True Value':<12} | {'Optimized Value'}")
    print("-" * 50)
    for name, true_val, opt_val in zip(config.PARAMS_TO_OPTIMIZE, config.TRUE_PARAMS, final_params):
        print(f"{name:<15} | {true_val:<12.6f} | {opt_val:.6f}")

if __name__ == "__main__":
    main()