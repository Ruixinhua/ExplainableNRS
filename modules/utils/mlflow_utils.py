import mlflow


def log_params(params, **kwargs):
    with mlflow.start_run(**kwargs):
        for key, value in params.items():
            try:
                mlflow.log_param(key, value)
            except mlflow.exceptions.MlflowException:
                pass


def log_metrics(metrics, **kwargs):
    with mlflow.start_run(**kwargs):
        for key, value in metrics.items():
            try:
                mlflow.log_metric(key, value)
            except mlflow.exceptions.MlflowException:
                pass


def set_experiment(experiment_name):
    return mlflow.set_experiment(experiment_name)
