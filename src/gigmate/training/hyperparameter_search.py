from clearml import Task
from clearml.automation import HyperParameterOptimizer
from clearml.automation.optuna import OptunaObjective
from clearml.automation.parameters import UniformIntegerParameterRange, UniformParameterRange
from clearml.automation.job import LocalClearmlJob
from gigmate.training.train import train_model
from gigmate.utils.constants import get_clearml_project_name, get_params

def objective(params):
    task = Task.init(
        project_name=get_clearml_project_name(),
        task_name='hyperparameter_optimization_subtask',
        task_type=Task.TaskTypes.training,
    )
    task.connect(params)

    # Train the model
    train_model(params)

    # Get the last reported value for val_accuracy
    best_val_accuracy = task.get_last_scalar_metrics()['val_accuracy']

    return best_val_accuracy

def optimize_hyperparameters():
    base_params = get_params()
    optimizer = HyperParameterOptimizer(
        base_task_id=None,
        hyper_parameters=[
            UniformIntegerParameterRange('epochs', min_value=10, max_value=50, step_size=10),
            UniformParameterRange('learning_rate', min_value=1e-5, max_value=1e-3, step_size=1e-5),
            UniformIntegerParameterRange('num_heads', min_value=4, max_value=16, step_size=4),
            UniformIntegerParameterRange('num_layers', min_value=2, max_value=12, step_size=2),
            UniformIntegerParameterRange('d_model', min_value=128, max_value=1024, step_size=128),
            UniformIntegerParameterRange('dff', min_value=512, max_value=4096, step_size=512),
            UniformParameterRange('dropout_rate', min_value=0.1, max_value=0.3, step_size=0.1),
        ],
        objective_metric_title='val_accuracy',
        objective_metric_series='val_accuracy',
        objective_metric_sign='max',
        max_number_of_concurrent_tasks=6,
        optimization_time_limit=12 * 60 * 60,
        compute_time_limit=2 * 60 * 60,
        total_max_jobs=10,
        optimizer_class=OptunaObjective,
        execution_queue='default',  # Queue name for agents to monitor
        job_class=LocalClearmlJob,
    )

    optimizer.set_base_configuration(base_params)

    optimizer.start_locally()
    optimizer.wait()
    optimizer.stop()

    print(optimizer.get_top_experiments())
    print(optimizer.get_best_task())

if __name__ == '__main__':
    Task.init(project_name=get_clearml_project_name(), task_name='hyperparameter_optimization_controller')
    optimize_hyperparameters()