import numpy as np
from tqdm import tqdm


def list_2d(list):
    return np.array(list)[:, None].tolist()


def loo(X, y, model_class, model_params_list, progress_bar=False, mask=None):
    if mask is None:
        mask = np.ones(X.shape[0], dtype=bool)
    res = []
    if len(np.array(model_params_list).shape) == 1:
        model_params_list = list_2d(model_params_list)

    for params in tqdm(model_params_list, disable=not progress_bar):
        model = model_class(*params)

        error_cnt = 0

        for idx in range(X.shape[0]):
            active_mask = mask & (np.arange(X.shape[0]) != idx)
            x_part = X[active_mask]
            y_part = y[active_mask]

            model.train(x_part, y_part)

            error_cnt += model.predict_class(X[idx]) != np.argmax(y[idx])
        error_cnt /= X.shape[0]
        res.append([params, error_cnt])
    return res
