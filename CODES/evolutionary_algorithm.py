# Functions 

## Dispersion curve estimative 

import numpy as np
import random
from CODES.modeling import calculate_parameters_from_vs
from CODES.dispersion_curves import estimate_disp_from_velocity_model

def create_velocity_model_from_profile_vs(model_profile):
    '''
    Parameters
    ----------
    Velocity model: 1-D numpy.array
        [thickness,velocity_p,velocity_s,density]
            - Layer thickness (in km).
            - Layer P-wave velocity (in km/s).
            - Layer S-wave velocity (in km/s).
            - Layer density (in g/cm3).     
    '''
    
    vmodel = []
    for (thickness,vs) in zip(*model_profile):
 
        vp, dens = calculate_parameters_from_vs(vs)
        
        # thickness, Vp, Vs, density
        # km, km/s, km/s, g/cm3

        vmodel.append([thickness,vp/1000,vs/1000,dens/1000])
   
    velocity_model = np.array(vmodel)    

    return velocity_model
    
# ---------------------------------------------------------------------

def create_layers(min_thick_layer, max_thick_layer,max_total=2.0):
    '''
    This function generates a list of random layer thicknesses that sum exactly to a specified total (`max_total`). 
    Each layer thickness is randomly drawn from a uniform distribution between `min_esp` and `max_esp`, 
    ensuring that all layers meet the minimum thickness requirement (`min_esp`). 
    The final layer is adjusted so that the total thickness precisely matches `max_total`.

    Parameters:
    -----------
    min_thick_layer : float
        Minimum thickness of an individual layer (m).
    max_thick_layer : float
        Maximum thickness of an individual layer  (m).
    max_total : float, optional (default=2 meters)
        Maximum total thickness of all layers (m).

    Returns:
    --------
    thick_lst : list of float
        A list of layer thicknesses.

    '''

    values = []
    current_sum = 0.0
    
    while True:
        remaining = round(max_total - current_sum, 2)
        
        possible_values = [v for v in [min_thick_layer+tk for tk in np.arange(0.01,max_thick_layer-min_thick_layer,0.01)] if max_thick_layer <= remaining]

        if not possible_values:
            values.append(remaining)
            break
        
        else:
            choice = random.choice(possible_values)
            values.append(round(choice,2))
            current_sum = round(current_sum + choice, 1)        
            
            if current_sum == max_total:
                break

    return values

# ------------------------------------------------------------------

def uniform(low_thick, up_thick,max_total,low_vels,up_vels):
    """
    Generates a random velocity model with layer thicknesses (m) and Vs values (m/s).

    This function first creates a set of random layer thicknesses using 
    `create_layers()`, then assigns shear wave velocities (Vs) to each layer 
    based on a uniform distribution.

    Parameters:
    -----------
    low_thick : float
        Lower bound for layer thickness (m).
    up_thick : float
        Upper bound for layer thickness (m).
    low_vels : float
        Lower bound for Vs values (m/s).
    up_vels : float
        Upper bound for Vs values (m/s).

    Returns:
    --------
    model : list
        A list containing:
        - `thickness_lst` (list of float): Layer thicknesses.
        - `vs_lst` (list of float): Corresponding Vs values.

    Notes:
    ------
    - The first layer's Vs is sampled uniformly between `low_vels` and `up_vels`.
    - Subsequent layers have Vs values increasing with depth, where the upper 
      and lower bounds for Vs are scaled by the layer index.
    - This function is used in the DEAP framework for generating models:
      
        toolbox.register("model", uniform, lower_thick, upper_thick, lower_vs, upper_vs)
    """
    
    thickness_lst = create_layers(low_thick,up_thick,max_total)
    
    vs_lst = []
    for s in range(1,len(thickness_lst)+1):
        if s == 1:
            vs_lst.append(round(np.random.uniform(low_vels,up_vels)))
        else:
            vs_lst.append(round(np.random.uniform(low_vels*(s),up_vels*(s))))
    return [thickness_lst,vs_lst]
# ------------------------------------------------------------------

def inversion_objective(individual, true_disp,number_samples=100):
    """
    Objective function for inversion using DEAP.

    This function evaluates the misfit between the experimental Rayleigh wave 
    dispersion data and the theoretical dispersion curve simulated from a given 
    shear wave velocity (Vs) profile.

    Parameters:
    -----------
    individual : list or array
        Estimated Vs profile used for optimization.
    true_disp : array
        Experimental Rayleigh wave phase velocity dispersion data.
    number_samples : int, optional (default=100)
        Number of frequency samples considered for misfit calculation.

    Returns:
    --------
    misfit : float
        Misfit value computed as the root mean square error (RMSE) 
        normalized by the standard deviation of the experimental data.
    
    Notes:
    ------
    - The theoretical dispersion curve is obtained by first creating a velocity 
      model from the Vs profile and then estimating the Rayleigh wave phase velocities.
    - The misfit formula is:
    
        misfit = sqrt(sum((xdi - xci)^2 / σi^2) / nf)
        
      where:
        - xdi: Experimental Rayleigh wave phase velocity at frequency fi
        - xci: Theoretical Rayleigh wave phase velocity for the trial model at fi
        - σi: Standard deviation of the experimental data at fi
        - nf: Number of frequency samples
    - If an error occurs, the function returns a high misfit value (10).
    """
    
    try:
        simulated_velocity_model = create_velocity_model_from_profile_vs(individual)
            
        simulated_cpr = estimate_disp_from_velocity_model(simulated_velocity_model)
            
        simulated_dispersion = simulated_cpr.velocity*1000
                
        nf = number_samples 
        sigma = np.std(true_disp)
    
        misfit = np.sqrt(np.sum(((true_disp - simulated_dispersion) ** 2) / (sigma ** 2)) / nf)

   
    except:
        misfit = 10
        
    return misfit,

# ------------------------------------------------------------------

def statistics_save(individual):
    """
    Retrieves the fitness value of an individual.

    This function returns the fitness value(s) of the given individual, 
    which is used for tracking statistics such as mean, standard deviation, 
    minimum, and maximum fitness during the optimization process.

    Parameters:
    -----------
    individual : object
        An individual solution with an assigned fitness value.

    Returns:
    --------
    fitness_value : tuple
        The fitness value(s) of the individual.
    """
    
    return individual.fitness.values

# ------------------------------------------------------------------

def mutate_gaussian(ind, mutpb=0.1):
    """
    Applies Gaussian mutation to the individual's values (layer thickness and velocity), 
    ensuring the structure of each sublist remains unchanged.

    Parameters:
    -----------
    ind : list
        The individual consisting of two sublists: thicknesses and velocities.
    mutpb : float, optional (default=0.1)
        Probability of mutating each value.

    Returns:
    --------
    tuple
        The mutated individual.
    """
    for i in range(len(ind)):  # Iterate over sublists (thickness and velocity)
        for j in range(len(ind[i])):  # Iterate over elements in sublist
            if random.random() < mutpb:  # Mutation probability check
                mu = ind[i][j] # Mean of the Gaussian distribution.
                sigma = mu*0.1 # Standard deviation of the Gaussian distribution.
                ind[i][j] += random.gauss(mu, sigma)  # Apply Gaussian noise
    return ind,

# ------------------------------------------------------------------

def crossover_two_point(ind1, ind2, cxpb=0.5):
    """
    Applies Two-Point Crossover to individuals while preserving the internal 
    structure of their sublists and respecting the shortest length between them.

    Parameters:
    -----------
    ind1 : list
        The first individual, consisting of two sublists.
    ind2 : list
        The second individual, also consisting of two sublists.
    cxpb : float, optional (default=0.5)
        The probability of performing crossover.

    Returns:
    --------
    tuple
        The two individuals after crossover.
    """
    for i in range(len(ind1)):  # Iterate over sublists (thickness and velocity)
        if random.random() < cxpb:  # Check if crossover occurs for this sublist
            # Determine the shortest length between corresponding sublists
            size = min(len(ind1[i]), len(ind2[i]))
            if size > 1:  # Perform crossover only if there are at least two elements
                # Select two crossover points
                point1, point2 = sorted(random.sample(range(size), 2))
                
                # Swap values between the two points
                for j in range(point1, point2 + 1):
                    ind1[i][j], ind2[i][j] = ind2[i][j], ind1[i][j]
                    
    return ind1, ind2

# ------------------------------------------------------------------

def configure_deap(estimated_disp,lower_thick,upper_thick,lower_vs,upper_vs):

    # Fitness and Individual Creation:
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    # Toolbox Initialization:
    toolbox = base.Toolbox()

    # Using Multiple Processors 
    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)
    
    # Attribute Generator:
    toolbox.register("model", uniform, lower_thick, upper_thick, lower_vs, upper_vs)

    # Individual and Population Initialization:
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.model)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Evaluation Function:
    toolbox.register("evaluate", inversion_objective, true_disp=estimated_disp)

    # Crossover Operation:
    toolbox.register("mate", crossover_two_point)

    # Mutation Operation:
    toolbox.register("mutate", mutate_gaussian)
    
    # Selection Strategy:
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    return toolbox