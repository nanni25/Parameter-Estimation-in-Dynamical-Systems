import roadrunner
import numpy as np
import config

rr = roadrunner.RoadRunner(config.MODEL_PATH)

# Force RoadRunner to strictly output only our target variables in the result matrix
rr.timeCourseSelections = config.MEAN_VARIABLES

def generate_synthetic_targets(true_params):
    rr.resetAll()
    
    for param_id, param_val in zip(config.PARAMS_TO_OPTIMIZE, true_params):
        rr.setValue(param_id, param_val)
        
    result = rr.simulate(0, config.SIMULATION_TIME, steps=config.SIMULATION_STEPS)
    
    # Return the target matrix directly instead of saving it globally
    return np.array(result)

def evaluate_loss(theta, target_data):
    rr.resetAll()
    
    actual_params = np.exp(theta)
    
    for param_id, param_val in zip(config.PARAMS_TO_OPTIMIZE, actual_params):
        rr.setValue(param_id, param_val)
        
    result = rr.simulate(0, config.SIMULATION_TIME, steps=config.SIMULATION_STEPS)
    simulated_y = np.array(result)

    epsilon = 1e-8
    # Calculate loss using the explicitly passed target_data
    loss = np.sum(((simulated_y - target_data) / (target_data + epsilon))**2)
    
    return loss