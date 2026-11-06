
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor


def run_algorithms():

    # Создаём тестовые данные
    X, y = make_regression(
        n_samples=200,
        n_features=1,
        noise=20,
        random_state=42
    )

    # Разделяем данные на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Создаём три алгоритма
    models = {
        "Decision Tree": DecisionTreeRegressor(random_state=42),

        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            random_state=42
        ),

        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100,
            random_state=42
        )
    }

    print("\nРезультаты работы моделей:")

    predictions = {}

    # Обучаем каждый алгоритм
    for name, model in models.items():

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        predictions[name] = y_pred

        print(f"\nАлгоритм: {name}")
        print("Первые 5 реальных значений:", y_test[:5])
        print("Первые 5 прогнозов:", y_pred[:5])

    return y_test, predictions
