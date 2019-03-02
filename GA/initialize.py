from skimage.transform import FundamentalMatrixTransform
import numpy as np


def initialize_matrix(icls,m,n,values,inliers):
    FM_Matrix = np.zeros((m,n))
    inliers = inliers
    for i in range(m):
        for j in range(n):
            FM_Matrix[i][j] = values[i][j]
    FM_Matrix = FundamentalMatrixTransform(matrix=FM_Matrix)
    return icls(FM_Matrix,inliers)