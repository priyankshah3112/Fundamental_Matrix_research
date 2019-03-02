import numpy as np
from deap import base
from deap import creator
from deap import tools
from GA.crossover import one_point_column_crossover,one_point_row_crossover,one_point_diagonal_crossover
from GA.initialize import initialize_matrix
from GA.fitness import residual
from GA.mutation import better_inlier_variants
from FM.FM_inliers import FundamentalMatrixInliers
from deap import algorithms

creator.create("FitnessMin", base.Fitness, weights=(1.0,))
creator.create("Individual", FundamentalMatrixInliers, fitness=creator.FitnessMin)



toolbox = base.Toolbox()
toolbox.register("individual", initialize_matrix , creator.Individual,3,3)
toolbox.register("mate",one_point_row_crossover)
toolbox.register("evaluate",residual)
toolbox.register("mutate",better_inlier_variants)
toolbox.register("select",algorithms.tools.selTournament,tournsize=3)


def ga(initial_population):

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    population = []
    for i in range(len(initial_population)):
        value_i = initial_population[i].FM_Matrix.params
        inliers_i = initial_population[i].inliers
        ind_i = toolbox.individual(values=value_i,inliers=inliers_i)
        population = population + [ind_i]

    algorithms.eaSimple(population=population, toolbox=toolbox, cxpb=0.8, mutpb=0.2, ngen=100, stats=stats)

    return population,stats







