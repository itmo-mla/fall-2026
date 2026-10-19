from data import load_iris_split, standardize
from validation import select_k_loo


def main():
    x_train, x_test, y_train, _, _, _ = load_iris_split()
    x_train, _, _, _ = standardize(x_train, x_test)
    k_values = range(1, 31)
    best_k, risks = select_k_loo(x_train, y_train, k_values)

    for k, risk in zip(k_values, risks):
        print(f"k={k:2d}, LOO-риск={risk:.4f}, ошибок={round(risk * len(y_train))}")
    print(f"Оптимальное k: {best_k}")
    print(f"Минимальный LOO-риск: {risks.min():.4f}")


if __name__ == "__main__":
    main()
