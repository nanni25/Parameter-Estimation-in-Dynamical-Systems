import numpy as np

def evaluate_loss(rr, theta, target_dict, params_to_optimize, mean_variables, sim_time, sim_steps):
    rr.resetAll()
    
    theta = np.clip(theta, -6.0, 6.0)
    actual_params = 10 ** theta
    
    for param_id, param_val in zip(params_to_optimize, actual_params):
        rr.setValue(param_id, param_val)
        
    result = rr.simulate(0, sim_time, steps=sim_steps)
    simulated_y = np.array(result)

    simulated_means = simulated_y[-1]
    
    target_values = []
    for var in mean_variables:
        species_id = var.replace("y_", "species_")
        target_values.append(target_dict[species_id])
        
    target_array = np.array(target_values)
    
    epsilon = 1e-8
    loss = np.sum(((simulated_means - target_array) / (target_array + epsilon))**2)
    
    return loss