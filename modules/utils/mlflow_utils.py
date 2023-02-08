import mlflow


def log_params(params):
    for key, value in params.items():
        try:
            mlflow.log_param(key, value)
        except mlflow.exceptions.MlflowException:
            pass


def log_metrics(metrics):
    for key, value in metrics.items():
        try:
            mlflow.log_metric(key, value)
        except mlflow.exceptions.MlflowException:
            pass


def set_experiment(experiment_name):
    return mlflow.set_experiment(experiment_name)


def get_experiment_id(experiment_name):
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
    except mlflow.exceptions.MlflowException:
        experiment = mlflow.create_experiment(experiment_name)
    return experiment.experiment_id
