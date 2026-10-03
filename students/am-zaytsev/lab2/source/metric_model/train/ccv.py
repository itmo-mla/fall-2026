import numpy as np


def get_m_closest_class(item, X, y, m):
    dist = np.linalg.norm(X - item, axis=1)

    sorted_idx_list = np.argsort(dist)[1:]
    y_sort = y[sorted_idx_list]
    return y_sort[m]


def compact_profile(X: np.array, y: np.array, m: int):
    profile_list = []
    for i in range(X.shape[0]):
        x = X[i]
        y_m = get_m_closest_class(x, X, y, m)
        profile_list.append(int(y_m != y[i]))
    return np.mean(profile_list)


if __name__ == "__main__":
    from plotly import express as px

    width = 10
    dist = 10
    n = 200
    x_pos = np.random.uniform(-(width + dist / 2), -(dist / 2), (n // 2, 1))
    y_pos = np.ones_like(x_pos) * -1

    x_neg = np.random.uniform(dist / 2, width + dist / 2, (n // 2, 1))
    y_neg = np.ones_like(x_pos)

    x = np.concat([x_pos, x_neg], 0)
    y = np.concat([y_pos, y_neg], 0)

    plot = px.scatter(x=x[:, 0], y=[0] * x.shape[0], color=list(map(str, y[:, 0])))
    plot.show(renderer="browser")

    m_list = np.arange(0, n - 1)
    p_list = [compact_profile(x, y, m) for m in m_list]
    plot = px.scatter(x=m_list, y=p_list)
    plot.show(renderer="browser")
