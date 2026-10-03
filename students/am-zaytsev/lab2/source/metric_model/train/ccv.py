from math import comb

import numpy as np

from metric_model.dataset.utils import remove_row


def get_sorted_m(arr, m):
    return np.quantile(arr, m / (arr.shape[0] - 1), method="nearest")


def get_m_closest_class(item, X, y, m, return_idx=False):
    dist = np.linalg.norm(X - item, axis=1)
    dist_m = get_sorted_m(dist, m)
    m_close_idx = np.argmin(np.abs(dist - dist_m))
    if not return_idx:
        return y[m_close_idx][0]
    return y[m_close_idx][0], m_close_idx


def compact_profile(X: np.array, y: np.array, m: int):
    profile_list = []
    for i in range(X.shape[0]):
        x = X[i]
        y_m = get_m_closest_class(x, X, y, m)
        p = int(y_m != y[i, 0])
        # if p == 1:
        #     print(X[i])
        profile_list.append(p)
    return np.mean(profile_list)


def ccv(X, y, k, l):
    """
    CCV(X^L) = sum_{m=1}^{k} Pi(m) * C(L-1, l-1-m) / C(L, l)
    l : ell  (control / test set size)
    k : number of terms in the sum
    """
    L = X.shape[0]
    denom = comb(L, l)

    ccv_sum = 0.0
    for m in range(1, k + 1):
        r = l - 1 - m
        if 0 <= r <= L - 1:
            num = comb(L - 1, r)
        else:
            num = 0
        ccv_sum += compact_profile(X, y, m) * num / denom

    return ccv_sum


def loo(X, y, k):
    return ccv(X, y, k, 1)


def _color(v):
    colors = ["red", "green", "blue", "brown"]
    return colors[hash(v) % len(colors)]


def draw_pm():
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # 1. Initialize a subplot grid (1 row, 2 columns)
    fig = make_subplots(rows=3, cols=2, subplot_titles=("Data", " П(m)"))

    n = 1000
    width = 10

    for row_i, dist in enumerate([10, 0, -10]):
        row_i += 1
        x_pos = np.random.uniform(-(width + dist / 2), -(dist / 2), (n // 2, 2))
        y_pos = np.ones_like(x_pos) * -1

        x_neg = np.random.uniform(dist / 2, width + dist / 2, (n // 2, 2))
        y_neg = np.ones_like(x_pos)

        x = np.concat([x_pos, x_neg], 0)
        y = np.concat([y_pos, y_neg], 0)
        x[:, 1] = np.random.uniform(
            -(width + dist / 2), (width + dist / 2), x[:, 1].shape
        )

        fig.add_trace(
            go.Scatter(
                x=x[:, 0],
                y=x[:, 1],
                mode="markers",
                marker=dict(color=list(map(_color, y[:, 0].tolist()))),
            ),
            row=row_i,
            col=1,
        )

        m_list = np.linspace(1, n - 1, 100)
        p_list = [compact_profile(x, y, m) for m in m_list]

        fig.add_trace(
            go.Scatter(x=m_list, mode="markers", y=p_list),
            row=row_i,
            col=2,
        )
        fig.update_xaxes(title_text="M Parameter", row=row_i, col=2)
        fig.update_yaxes(
            title_text="Profile Score", range=[-0.05, 1.05], row=row_i, col=2
        )

    fig.show(renderer="browser")
    # fig.write_image("images/compact_profile_plots.jpg")


def get_k1_k2_list(x):
    k1_idx_list = []
    k2_idx_list = []
    for i in range(x.shape[0]):
        k1_idx, k2_idx = closest_idx(x, x[i], 2)
        k1_idx_list.append(int(k1_idx))
        k2_idx_list.append(int(k2_idx))
    return k1_idx_list, k2_idx_list


def remove_row_compact_profile(x, y, row_i, old_compact_profile):
    l = x.shape[0]
    k1_idx_list, k2_idx_list = get_k1_k2_list(x)
    lp_new = old_compact_profile * l

    lp_new = lp_new - int(y[k1_idx_list[row_i]][0] != y[row_i][0])

    for i in range(x.shape[0]):
        k1_idx = k1_idx_list[i]
        k2_idx = k2_idx_list[i]

        if k1_idx != row_i:
            continue

        if (y[i] == y[row_i]) and (y[i] != y[k2_idx]):
            lp_new += 1
        if (y[i] != y[row_i]) and (y[i] == y[k2_idx]):
            lp_new -= 1
    return lp_new / (l - 1)


def closest_idx(x, xi, n=None):
    if n is None:
        n = x.shape[0] - 1

    dist = np.linalg.norm(x - xi, axis=1)
    return np.argsort(dist)[1 : n + 1]


if __name__ == "__main__":
    a = remove_row
    x = np.float32([0, 1, 3, 7, 20]).reshape(-1, 1)
    y = np.float32([0, 0, 1, 0, 0]).reshape(-1, 1)

    p_old = compact_profile(x, y, 1)

    L = x.shape[0]

    for del_i in range(x.shape[0]):
        p_new_pred = remove_row_compact_profile(x, y, del_i, p_old)

        p_new_true = compact_profile(remove_row(x, del_i), remove_row(y, del_i), 1)
        assert p_new_pred == p_new_true
        print(del_i, p_new_true)
