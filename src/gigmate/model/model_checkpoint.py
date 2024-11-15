from typing import Optional, cast
from clearml import Task
from gigmate.utils.constants import get_clearml_project_name

LATEST_TASK_CHECKPOINT_ID = None  # '5af742b55f8f4786a5356a99c3f4943e'
ARTIFACT_NAME = 'weights-epoch-7'


def get_task(task_id: Optional[str] = LATEST_TASK_CHECKPOINT_ID) -> Optional[Task]:
    if task_id is None:
        return None

    return Task.get_task(task_id=task_id, project_name=get_clearml_project_name())


def get_task_artifact(task_id: str, artifact_name: str):
    task = cast(Task, get_task(task_id))
    return task.artifacts[artifact_name].get_local_copy()


def get_latest_model_checkpoint_path(task_id: Optional[str] = LATEST_TASK_CHECKPOINT_ID, artifact_name: str = ARTIFACT_NAME):
    if task_id is None:
        return None
    
    return get_task_artifact(task_id, artifact_name)
