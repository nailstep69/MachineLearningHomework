
import numpy as np


def calculate_metrics(y_true, y_pred):

    # Преобразуем данные в массивы NumPy
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # MAE - средняя абсолютная ошибка
    mae = np.mean(np.abs(y_true - y_pred))

    # MSE - средняя квадратичная ошибка
    mse = np.mean((y_true - y_pred) ** 2)

    # RMSE - корень из средней квадратичной ошибки
    rmse = np.sqrt(mse)

    # R2 - коэффициент детерминации
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    r2 = 1 - (ss_res / ss_tot)

    return mae, mse, rmse, r2


def run_metrics(y_test, predictions):
    print("\nОценка точности моделей:")



    for name, y_pred in predictions.items():

        mae, mse, rmse, r2 = calculate_metrics(
            y_test, y_pred
        )

        print(f"\nАлгоритм: {name}")
        print(f"MAE:  {mae:.2f}")
        print(f"MSE:  {mse:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"R2:   {r2:.4f}")
