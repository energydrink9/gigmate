
from clearml import Task
from gigmate.constants import get_clearml_project_name

LATEST_TASK_CHECKPOINT_ID = '55380192b888402180da01991a584f52'

def get_task_artifact(task_id: str, artifact_name: str):
    preprocess_task = Task.get_task(task_id=task_id, project_name=get_clearml_project_name())
    return preprocess_task.artifacts[artifact_name].get_local_copy()

def get_latest_model_checkpoint_path():
    return get_task_artifact(LATEST_TASK_CHECKPOINT_ID, 'weights-epoch-6')
