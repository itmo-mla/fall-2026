from sklearn.linear_model import RidgeClassifier

from src.dataset import DataSet
from src.metrics import confusion_matrix


def main():
    ds = DataSet()

    model = RidgeClassifier()
    model.fit(ds.train.X, ds.train.Y)
    print('Weights after:', model.coef_, model.intercept_)

    Y_pred = model.predict(ds.test.X)
    tp, tn, fp, fn = confusion_matrix(ds.test.Y.to_numpy(), Y_pred)
    print(tp, tn, fp, fn)
    
    print('Accuracy:', (tp + tn) / (tp + tn + fp + fn))
    print('Precision:', tp / (tp + fp))

if __name__ == '__main__':
    main()
