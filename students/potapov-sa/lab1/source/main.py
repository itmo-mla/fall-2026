from sklearn.linear_model import RidgeClassifier

from src.dataset import DataSet
from src.linear_classificator import LinearClassificator
from src.metrics import confusion_matrix
from src.plot import plot_margins


def implemented_solution(ds: DataSet):
    print('### MY IMPLEMENTATION ###\n')

    model = LinearClassificator(ds.feature_count)
        
    # print('Weights before:', model.weights, end='\n\n')
    model.gradient_descent(ds.train)
    # print('Weights after:', model.weights, end='\n\n')

    for predict_sample in ds.train, ds.test:
        Y_pred = model.predict(predict_sample)
        tp, tn, fp, fn = confusion_matrix(predict_sample.Y.to_numpy(), Y_pred)
        print(tp, tn, fp, fn)

        print(f'{predict_sample.get_name()} Accuracy:', (tp + tn) / (tp + tn + fp + fn))
        print(f'{predict_sample.get_name()} Precision:', tp / (tp + fp))

    print('\n#########################')


def reference_solution(ds: DataSet):
    print('\n### REFERENCE IMPLEMENTATION ###\n')

    model = RidgeClassifier()
    model.fit(ds.train.X, ds.train.Y)
    print('Weights after:', model.coef_, model.intercept_)

    Y_pred = model.predict(ds.test.X)
    tp, tn, fp, fn = confusion_matrix(ds.test.Y.to_numpy(), Y_pred)
    print(tp, tn, fp, fn)
    
    print('Accuracy:', (tp + tn) / (tp + tn + fp + fn))
    print('Precision:', tp / (tp + fp))

    print('\n################################')


def main():
    ds = DataSet()
    implemented_solution(ds)
    reference_solution(ds)


if __name__ == '__main__':
    main()
