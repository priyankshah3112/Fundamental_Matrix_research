from deap import base
from deap import creator
from FM.FM_inliers import FundamentalMatrixInliers
from GA.initialize import initialize_matrix
from GA.fitness import inliers,residual
from skimage.transform import FundamentalMatrixTransform
creator.create("Individual", FundamentalMatrixInliers, fitness=creator.FitnessMax)


toolbox = base.Toolbox()
toolbox.register("individual", initialize_matrix , creator.Individual,3,3)
toolbox.register("evaluate",residual)

def make_all_permuations(Matrix):
    epsilon = [-2,-1,0,1,2]
    mutated_matrices = []
    for i in range(0, 3):
        for j in range(0, 3):
            for e in epsilon:
                new_matrix = Matrix.copy()
                new_matrix[i][j] = Matrix[i][j] + e * 0.01*Matrix[i][j]
                mutated_matrices.append(FundamentalMatrixTransform(new_matrix))
    return mutated_matrices


def better_inlier_variants(individual):
    Matrix = individual.FM_Matrix.params
    mutated_matrices = make_all_permuations(Matrix)
    mutated_population = []
    for FM in mutated_matrices:
        inliers_test = inliers(FM)
        ind_test = toolbox.individual(values = FM.params,inliers = inliers_test)
        mutated_population.append(ind_test)

    sorted_population = sorted(mutated_population,key=lambda x:x.inliers)
    return sorted_population[0],









