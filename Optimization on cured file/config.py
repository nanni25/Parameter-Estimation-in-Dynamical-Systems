import numpy as np

# 1. FILE PATH
MODEL_PATH = "models/R-HSA-5653890.sbml"

# 2. SBML PARAMETERS TO OPTIMIZE
PARAMS_TO_OPTIMIZE = [
    "lambda_5653873", "lambda_5653886", "lambda_5653878", 
    "K_in_70113", "K_in_735682", "K_in_528072",           
    "K_in_5653896", "K_in_5653881", "K_in_5653875",       
    "K_out_5653871", "K_out_205687", "K_out_425969"       
]

# 3. TARGET VALUES
MEAN_VARIABLES = [
    "y_5653871", "y_205687", "y_425969", "y_964746", "y_735682",
    "y_70113", "y_528072", "y_5653896", "y_5653885", "y_5653881", "y_5653875"
]

# Synthetic Experiment Variables
TRUE_PARAMS = np.array([1.5, 0.8, 1.2, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5])
TARGET_M = None

# 4. SIMULATION SETTINGS
SIMULATION_TIME = 100.0  
SIMULATION_STEPS = 100   

# 5. ES HYPERPARAMETERS
POPULATION_SIZE = 100      
NUM_GENERATIONS = 500      
LEARNING_RATE = 0.1       
SIGMA = 0.1