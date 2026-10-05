from sklearn.datasets import fetch_openml


def data_load():
    data = fetch_openml(data_id=54, as_frame=True)
    X = data.data
    y = data.target.to_numpy()
    return X, y
