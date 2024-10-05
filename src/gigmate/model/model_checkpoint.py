from typing import Optional
from clearml import Task
from gigmate.utils.constants import get_clearml_project_name

LATEST_TASK_CHECKPOINT_ID = None

def get_task_artifact(task_id: str, artifact_name: str):
    task = Task.get_task(task_id=task_id, project_name=get_clearml_project_name())
    return task.artifacts[artifact_name].get_local_copy()

def get_latest_model_checkpoint_path(task_id: Optional[str]=LATEST_TASK_CHECKPOINT_ID, artifact_name: str = 'weights-epoch-1'):
    if task_id is None:
        return None
    
    return get_task_artifact(task_id, artifact_name)
