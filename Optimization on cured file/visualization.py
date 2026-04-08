import numpy as np
import matplotlib.pyplot as plt
import roadrunner
import config

def plot_results(history_best_loss, history_mean_loss, species_names, target_values, simulated_means):
    # 1. Convergence Plot
    plt.figure(figsize=(10, 6))
    plt.plot(history_mean_loss, label='Mean Loss (Population)', color='#1f77b4', alpha=0.8)
    plt.plot(history_best_loss, label='Best Loss (Individual)', color='#d62728', linewidth=2)
    plt.yscale('log')
    plt.title('Optimizer Convergence over Generations', fontsize=14, fontweight='bold')
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Mean Squared Error (Log Scale)', fontsize=12)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('convergence_plot.pdf', dpi=300)
    plt.show()

    # 2. Target vs. Simulated Mean Bar Chart
    plt.figure(figsize=(14, 7))
    x = np.arange(len(species_names))  
    width = 0.35               
    
    fig, ax = plt.subplots(figsize=(14, 7))
    rects1 = ax.bar(x - width/2, target_values, width, label='Targets', color='#2ca02c')
    rects2 = ax.bar(x + width/2, simulated_means, width, label='Simulated Means', color='#ff7f0e')

    ax.set_ylabel('Concentration Value', fontsize=12)
    ax.set_title('Comparison of Targets vs. Mean Concentrations', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(species_names, rotation=45, ha='right', fontsize=10)
    ax.legend(fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('target_comparison.pdf', dpi=300)
    plt.show()

def plot_timeline(final_params, targets):
    print("\nGenerating Time-Course Timeline Plot...")
    
    # 1. Create a fresh roadrunner instance for the timeline
    rr = roadrunner.RoadRunner(config.MODEL_PATH)
    
    # 2. We want to plot the actual species over time, not the running means
    species_ids = [var.replace("y_", "species_") for var in config.MEAN_VARIABLES]
    rr.timeCourseSelections = ['time'] + species_ids
    
    # 3. Inject the optimized parameters
    rr.resetAll()
    for param_id, param_val in zip(config.PARAMS_TO_OPTIMIZE, final_params):
        rr.setValue(param_id, param_val)
        
    # 4. Simulate (we use more steps here for a smoother curve)
    result = rr.simulate(0, config.SIMULATION_TIME, steps=config.SIMULATION_STEPS * 5)
    
    # 5. Plotting
    plt.figure(figsize=(14, 8))
    time_array = result[:, 0]
    
    # Plot each species curve
    for i, sp_id in enumerate(species_ids):
        color = plt.cm.tab20(i % 20) # Get a distinct color for each species
        
        # Plot the simulated dynamic curve
        plt.plot(time_array, result[:, i+1], label=f"{sp_id} (Sim)", color=color, linewidth=2)
        
        # Plot the LLM target as a dashed line to see if it reaches it
        if sp_id in targets:
            plt.axhline(y=targets[sp_id], color=color, linestyle='--', alpha=0.5)

    plt.title('Time-Course Simulation with Optimized Parameters', fontsize=16, fontweight='bold')
    plt.xlabel('Time (Simulation Units)', fontsize=12)
    plt.ylabel('Concentration', fontsize=12)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    
    # Put legend outside the plot so it doesn't cover the lines
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., fontsize=9)
    plt.tight_layout()
    
    plt.savefig('timeline_plot.pdf', dpi=300, bbox_inches='tight')
    print("Saved timeline plot as 'timeline_plot.pdf'")
    plt.show()