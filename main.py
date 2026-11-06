from algorithms import run_algorithms
from metrics import run_metrics


def main():
    y_test, predictions = run_algorithms()

    run_metrics(y_test, predictions)


if __name__ == "__main__":
    main()