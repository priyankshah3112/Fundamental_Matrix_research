from skimage.transform import FundamentalMatrixTransform
class FundamentalMatrixInliers(object):
    FM_Matrix = FundamentalMatrixTransform()
    inliers = 0

    # The class "constructor" - It's actually an initializer
    def __init__(self, FM, inliers):
        self.FM_Matrix = FM
        self.inliers = inliers

