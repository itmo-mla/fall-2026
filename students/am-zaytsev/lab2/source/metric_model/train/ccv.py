import numpy as np


def get_sorted_m(arr, m):
    return np.quantile(arr, m / (arr.shape[0] - 1), method="nearest")


def get_m_closest_class(item, X, y, m):
    dist = np.linalg.norm(X - item, axis=1)
    dist_m = get_sorted_m(dist, m)
    m_close_idx = np.argmin(np.abs(dist - dist_m))

    return y[m_close_idx][0], m_close_idx


def compact_profile(X: np.array, y: np.array, m: int):
    profile_list = []
    for i in range(X.shape[0]):
        x = X[i]
        y_m = get_m_closest_class(x, X, y, m)
        # print(y_m, y[i])
        profile_list.append(int(y_m != y[i, 0]))
    return np.mean(profile_list)


if __name__ == "__main__":
    from plotly import express as px

    width = 10
    dist = 10
    n = 10
    x_pos = np.random.uniform(-(width + dist / 2), -(dist / 2), (n // 2, 1))
    y_pos = np.ones_like(x_pos) * -1

    x_neg = np.random.uniform(dist / 2, width + dist / 2, (n // 2, 1))
    y_neg = np.ones_like(x_pos)

    x = np.concat([x_pos, x_neg], 0)
    y = np.concat([y_pos, y_neg], 0)

    # plot = px.scatter(x=x[:, 0], y=[0] * x.shape[0], color=list(map(str, y[:, 0])))
    # plot.show(renderer="browser")

    m_list = np.arange(1, n - 1)
    p_list = [compact_profile(x, y, m) for m in m_list]
    plot = px.scatter(x=m_list, y=p_list)
    plot.show(renderer="browser")
