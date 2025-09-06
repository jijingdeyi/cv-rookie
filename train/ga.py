import pygad
import numpy as np


# GA
numberGeneration = 5  
numberParentsMating = 5
solutionPerPopulation = 5  
parents = -1

numberGenes = 7  
geneType = float  

minValue = 1  
maxValue = 5  

selectionType = 'rws' 

crossoverType = 'single_point' 
crossoverRate = 0.25 

mutationType = 'random' 
mutationReplacement = True  
mutationRate = 10  

mean_mse_loss_V = 1
mean_mse_loss_I = 1
mean_loss_decomp = 1
mean_fusionloss = 1
mean_box_loss = 1
mean_cls_loss = 1
mean_dfl_loss = 1

def fitnessFunction(geneticAlgorithm, solution, solution_idx):

    # weight normalization and limit the range of weight between 1 and 10
    for i in range(len(solution)):
        if solution[i] < 1:
            solution[i] = 1
        elif solution[i] > 10:
            solution[i] = 10

    # weight normalization and make the sum of weight equal to 10
    coo_sum = sum(solution)
    solution[0] = solution[0] / coo_sum * 10.
    solution[1] = solution[1] / coo_sum * 10.
    solution[2] = solution[2] / coo_sum * 10.
    solution[3] = solution[3] / coo_sum * 10.

    solution[4] = solution[4] / coo_sum * 10.
    solution[5] = solution[5] / coo_sum * 10.
    solution[6] = solution[6] / coo_sum * 10.

    outputExpected = solution[0] * mean_mse_loss_V + solution[1] * mean_mse_loss_I \
                        + solution[2] * mean_loss_decomp + solution[3] * mean_fusionloss\
                        + solution[4] * mean_box_loss + solution[5] * mean_cls_loss + solution[6] * mean_dfl_loss
    outputExpected = outputExpected.cpu()
    outputExpected = outputExpected.detach().numpy()

    fitnessValue = 1 / (np.abs(outputExpected) + 0.000001)
    return fitnessValue

geneticAlgorithm = pygad.GA(
    num_generations=numberGeneration,
    num_parents_mating=numberParentsMating,
    num_genes=numberGenes,
    gene_type=geneType,
    fitness_func=fitnessFunction,

    sol_per_pop=solutionPerPopulation,
    init_range_high=maxValue,
    init_range_low=minValue,

    parent_selection_type=selectionType, # rolewheel selection
    keep_parents=parents,
    crossover_type=crossoverType,

    mutation_type=mutationType,
    mutation_by_replacement=mutationReplacement,
    random_mutation_max_val=maxValue,
    random_mutation_min_val=minValue,
    mutation_percent_genes=mutationRate,

    save_solutions=True,
    save_best_solutions=False,  # 拒绝保存best_solutions
    suppress_warnings=True,
)