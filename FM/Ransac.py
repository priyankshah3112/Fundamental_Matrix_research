import numpy as np
import numbers


def _dynamic_max_trials(n_inliers, n_samples, min_samples, probability):

    if n_inliers == 0:
        return np.inf

    nom = 1 - probability
    if nom == 0:
        return np.inf

    inlier_ratio = n_inliers / float(n_samples)
    denom = 1 - inlier_ratio ** min_samples
    if denom == 0:
        return 1
    elif denom == 1:
        return np.inf

    nom = np.log(nom)
    denom = np.log(denom)
    if denom == 0:
        return 0

    return int(np.ceil(nom / denom))

def check_random_state(seed):

    # Function originally from scikit-learn's module sklearn.utils.validation
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)

def ransac(data, model_class, min_samples, residual_threshold,
           is_data_valid=None, is_model_valid=None,
           max_trials=100, stop_sample_num=np.inf, stop_residuals_sum=0,
           stop_probability=1, random_state=None):


    best_model = None
    best_inlier_num = 0
    best_inlier_residuals_sum = np.inf
    best_inliers = None

    random_state = check_random_state(random_state)

    if min_samples < 0:
        raise ValueError("`min_samples` must be greater than zero")

    if max_trials < 0:
        raise ValueError("`max_trials` must be greater than zero")

    if stop_probability < 0 or stop_probability > 1:
        raise ValueError("`stop_probability` must be in range [0, 1]")

    if not isinstance(data, list) and not isinstance(data, tuple):
        data = [data]

    # make sure data is list and not tuple, so it can be modified below
    data = list(data)
    # number of samples
    num_samples = data[0].shape[0]
    fundaMatrix_and_inliers=[]
    for num_trials in range(max_trials):

        # choose random sample set
        samples = []
        random_idxs = random_state.randint(0, num_samples, min_samples)
        for d in data:
            samples.append(d[random_idxs])

        # check if random sample set is valid
        if is_data_valid is not None and not is_data_valid(*samples):
            continue

        # estimate model for current random sample set
        sample_model = model_class()

        success = sample_model.estimate(*samples)

        if success is not None:  # backwards compatibility
            if not success:
                continue

        # check if estimated model is valid
        if is_model_valid is not None \
                and not is_model_valid(sample_model, *samples):
            continue

        sample_model_residuals = np.abs(sample_model.residuals(*data))
        # consensus set / inliers
        sample_model_inliers = sample_model_residuals < residual_threshold
        sample_model_residuals_sum = np.sum(sample_model_residuals**2)

        # choose as new best model if number of inliers is maximal
        sample_inlier_num = np.sum(sample_model_inliers)
        fundaMatrix_and_inliers.append([sample_model.params,sample_inlier_num])
        if (
            # more inliers
            sample_inlier_num > best_inlier_num
            # same number of inliers but less "error" in terms of residuals
            or (sample_inlier_num == best_inlier_num
                and sample_model_residuals_sum < best_inlier_residuals_sum)
        ):
            best_model = sample_model
            best_inlier_num = sample_inlier_num
            best_inlier_residuals_sum = sample_model_residuals_sum
            best_inliers = sample_model_inliers
            if (
                best_inlier_num >= stop_sample_num
                or best_inlier_residuals_sum <= stop_residuals_sum
                or num_trials
                    >= _dynamic_max_trials(best_inlier_num, num_samples,
                                           min_samples, stop_probability)
            ):
                break

    # estimate final model using all inliers
    if best_inliers is not None:
        # select inliers for each data array
        for i in range(len(data)):
            data[i] = data[i][best_inliers]
        best_model.estimate(*data)

    return best_model, best_inliers,fundaMatrix_and_inliers