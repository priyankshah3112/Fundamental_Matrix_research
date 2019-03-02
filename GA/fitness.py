import numpy as np
from deap import base
from deap import creator
from FM.FM_inliers import FundamentalMatrixInliers
from GA.initialize import initialize_matrix
from helper import convert_json_to_numpy
from config import residual_threshold
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", FundamentalMatrixInliers, fitness=creator.FitnessMax)


toolbox = base.Toolbox()
toolbox.register("individual", initialize_matrix , creator.Individual,3,3)

matches = convert_json_to_numpy('match_1'),convert_json_to_numpy('match_2')
def residual(individual):
    FM = individual.FM_Matrix
    FM_residuals = np.abs(FM.residuals(*matches))
    FM_residuals_sum = np.sum(FM_residuals ** 2)
    FM_inliers = (FM_residuals < residual_threshold).sum()
    return FM_inliers,

def inliers(FM):
    FM_residuals = np.abs(FM.residuals(*matches))
    FM_inliers = (FM_residuals < residual_threshold).sum()
    return FM_inliers



