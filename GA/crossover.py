import numpy as np
from deap import base
from deap import creator
from FM.FM_inliers import FundamentalMatrixInliers
from GA.initialize import initialize_matrix
from GA.fitness import inliers
from skimage.transform import FundamentalMatrixTransform
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", FundamentalMatrixInliers, fitness=creator.FitnessMax)
toolbox = base.Toolbox()
toolbox.register("individual", initialize_matrix , creator.Individual,3,3)

def one_point_row_crossover(ind1, ind2):
    cx = np.random.randint(0, 2) # selects a random value from [0,1,2]
    ind1 = ind1.FM_Matrix.params.copy()
    ind2 = ind2.FM_Matrix.params.copy()
    temp_ind1 = ind1.copy()
    temp_ind2 = ind2.copy()
    # single for loop takes care of all three cases where row == 0,1,2 (cx values) are selected for crossover
    for i in range(0,3):
        ind1[cx][i] = temp_ind2[cx][i]
        ind2[cx][i] = temp_ind1[cx][i]
    ind_new_1 = FundamentalMatrixTransform(ind1)
    ind_new_2 = FundamentalMatrixTransform(ind2)
    inliers_1 = inliers(ind_new_1)
    inliers_2 = inliers(ind_new_2)
    return toolbox.individual(values=ind1, inliers=inliers_1), toolbox.individual(values=ind2, inliers=inliers_2)

def one_point_diagonal_crossover(ind1,ind2):
    cx = np.random.randint(0, 1) # 0 for left diagonal and 1 for right diagonal
    ind1 = ind1.FM_Matrix.params.copy()
    ind2 = ind2.FM_Matrix.params.copy()
    temp_ind1 = ind1.copy()
    temp_ind2 = ind2.copy()
    matrix_size=3
    if cx == 1:
        for i in range(0, 3):
            ind1[i][i] = temp_ind2[i][i]
            ind2[i][i] = temp_ind1[i][i]
    elif cx ==0:
        for i in range(0, 3):
            ind1[matrix_size-1-i][i] = temp_ind2[matrix_size-1-i][i]
            ind2[matrix_size-1-i][i] = temp_ind1[matrix_size-1-i][i]
    ind_new_1 = FundamentalMatrixTransform(ind1)
    ind_new_2 = FundamentalMatrixTransform(ind2)
    inliers_1 = inliers(ind_new_1)
    inliers_2 = inliers(ind_new_2)
    return toolbox.individual(values=ind1, inliers=inliers_1), toolbox.individual(values=ind2, inliers=inliers_2)


def one_point_column_crossover(ind1, ind2):
    cx = np.random.randint(0,2)
    ind1 = ind1.FM_Matrix.params.copy()
    ind2 = ind2.FM_Matrix.params.copy()
    temp_ind1 = ind1.copy()
    temp_ind2 = ind2.copy()
    if cx == 1:
        for i in range(cx + 1, cx + 2):
            for j in range(0,3):
                ind1[j][i] = temp_ind2[j][i]
                ind2[j][i] = temp_ind1[j][i]
    elif cx == 0:
        for i in range(cx + 1):
            for j in range(0, 3):
                ind1[j][i] = temp_ind2[j][i]
                ind2[j][i] = temp_ind1[j][i]
    ind_new_1 = FundamentalMatrixTransform(ind1)
    ind_new_2 = FundamentalMatrixTransform(ind2)
    inliers_1 = inliers(ind_new_1)
    inliers_2 = inliers(ind_new_2)
    return toolbox.individual(values=ind1,inliers=inliers_1),toolbox.individual(values=ind2,inliers=inliers_2)