# conda install -c menpo opencv
import numpy as np
from skimage.measure import ransac
import cv2
from skimage.transform import FundamentalMatrixTransform
import os
from FM.Ransac import ransac
from FM.FM_inliers import FundamentalMatrixInliers
from helper import write_numpy_to_json
import config
global match1,match2
def get_image(directory,image_name):
    path = os.path.join(os.getcwd(),directory,image_name)
    return cv2.imread(path)


def save_image(directory, image_name,image):
    path = os.path.join(os.getcwd(),directory,image_name)
    cv2.imwrite(path,image)



def find_matches(image_1,image_2):

    image_1 = get_image('teddy',image_1 + '.png')
    image_2 = get_image('teddy',image_2 + '.png')
    sift=cv2.xfeatures2d.SIFT_create()
    gray_1= cv2.cvtColor(image_1,cv2.COLOR_BGR2GRAY)
    gray_2= cv2.cvtColor(image_2,cv2.COLOR_BGR2GRAY)
    keypoints_1,descriptors_1 = sift.detectAndCompute(gray_1,None)
    keypoints_2,descriptors_2 = sift.detectAndCompute(gray_2,None)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(descriptors_1,descriptors_2,k=2)
    good_matches = []
    match_1 = []
    match_2 = []
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.8*n.distance:
            good_matches.append(m)
            match_1.append(keypoints_1[m.trainIdx].pt)
            match_2.append(keypoints_2[m.queryIdx].pt)
    match_1 = np.int32(match_1)
    match_2 = np.int32(match_2)
    return match_1,match_2



def modify_FM(FM,n,matches):
    FM_with_error=[]
    FM_Matrix_original = FM.FM_Matrix.params
    epsilon_array = [i for i in range(-1*n, n+1)]
    epsilon_array = np.array(epsilon_array)
    epsilon_matrix = np.zeros((3,3,7))
    for a in range(0, 3):
        for b in range(0, 3):
            for i in range(0, 7):
                epsilon_matrix[a][b][i] = epsilon_array[i]

    while(True):
        FM_Matrix_with_errors_k=np.zeros((3,3))
        for a in range(0,3):
            for b in range(0,3):
                random_epsilon = epsilon_matrix[a][b][np.random.randint(0,7)]
                FM_Matrix_with_errors_k[a][b] = FM_Matrix_original[a][b]*(1+0.001*random_epsilon)
        FM_with_errors_k = FundamentalMatrixTransform(matrix=FM_Matrix_with_errors_k)
        inliers_k = check_inliers(matches, FM_with_errors_k)
        if inliers_k <= FM.inliers:
            continue
        else :
            ##TODO Equality check
            FM_with_error_inliers = FundamentalMatrixInliers(FM_with_errors_k,inliers_k)
            FM_with_error.append(FM_with_error_inliers)
        if len(FM_with_error)==20:
            break
    return FM_with_error


def check_inliers(matches,FM ,residual_threshold=config.residual_threshold):
    residuals = FM.residuals(*matches)
    inliers = (residuals < residual_threshold).sum()
    return inliers

def make_initial_population(n_fundamental_matrices,matches):
    model, inliers,FundaMatrix_and_inliers_list = ransac(matches,
                            FundamentalMatrixTransform, min_samples=8,
                            residual_threshold=config.residual_threshold, max_trials=n_fundamental_matrices)
    FM_inlier_list = []
    sorted_FM = sorted(FundaMatrix_and_inliers_list, key=lambda x: int(x[1]), reverse=True)
    top_10_FM = sorted_FM[:10]
    for item in top_10_FM:
        FM = FundamentalMatrixTransform(matrix=item[0])
        FM_inlier = FundamentalMatrixInliers(FM,item[1])
        FM_inlier_list.append(FM_inlier)
    epsiloned_FM = []

    for FM in FM_inlier_list:
        epsiloned_FM = epsiloned_FM + modify_FM(FM, 3,matches)

    return epsiloned_FM


def fm():
    match_1, match_2 = find_matches('image_1','image_2')
    write_numpy_to_json(match_1,'match_1')
    write_numpy_to_json(match_2,'match_2')
    population = make_initial_population(20,(match_1,match_2))
    return population






