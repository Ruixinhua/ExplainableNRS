import mlflow


def log_params(params, **kwargs):
    with mlflow.start_run(**kwargs):
        for key, value in params.items():
            mlflow.log_param(key, value)


def log_metrics(metrics, **kwargs):
    with mlflow.start_run(**kwargs):
        for key, value in metrics.items():
            mlflow.log_metric(key, value)


def set_experiment(experiment_name):
    return mlflow.set_experiment(experiment_name)
