import roadrunner
import numpy as np
import config

rr = roadrunner.RoadRunner(config.MODEL_PATH)
rr.timeCourseSelections = config.MEAN_VARIABLES

def evaluate_loss(theta, target_dict):

    rr.resetAll()
    
    theta = np.clip(theta, -6.0, 6.0)
    actual_params = 10 ** theta
    
    for param_id, param_val in zip(config.PARAMS_TO_OPTIMIZE, actual_params):
        rr.setValue(param_id, param_val)
        
    result = rr.simulate(0, config.SIMULATION_TIME, steps=config.SIMULATION_STEPS)
    simulated_y = np.array(result)

    # Calculate the mean of the simulation over time
    simulated_means = np.mean(simulated_y, axis=0)
    
    # Extract the target values from the dictionary
    target_values = []
    for var in config.MEAN_VARIABLES:
        species_id = var.replace("y_", "species_")
        target_values.append(target_dict[species_id])
        
    target_array = np.array(target_values)
    
    epsilon = 1e-8
    loss = np.sum(((simulated_means - target_array) / (target_array + epsilon))**2)
    
    return loss