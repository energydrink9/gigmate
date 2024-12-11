import os
import tempfile
import traceback
from typing import Optional, cast
from clearml import Task
from s3fs.core import S3FileSystem
from functools import lru_cache

from gigmate.utils.constants import get_clearml_project_name

# Set to none to start training from scratch, otherwise use checkpoint id to continue training from last checkpoint.
LATEST_TASK_CHECKPOINT_ID = None  # '24b6e0fbfdeb480ab1e652d64df3d0a6'  # '37b6cfb8f13c4d79b37810979683d3b5'  # 'e3b53c03bad94c8cb77fe8499300356a'  # 'cfbaa68397fd4113b21b0214fc16eb73'
EPOCH = 19  # 24  # 24  # 39
S3_CHECKPOINTS_STORAGE = True
CHECKPOINTS_BUCKET = 'gigmate-checkpoints'
OUTPUT_DIRECTORY = '/data' if os.path.exists('/data') else tempfile.gettempdir()
CHECKPOINTS_DIR = os.path.join(OUTPUT_DIRECTORY, 'checkpoints')
LATEST_CHECKPOINT_PATH = os.path.join(CHECKPOINTS_DIR, 'latest.ckpt')
LATEST_CHECKPOINT_EPOCH_PATH = os.path.join(CHECKPOINTS_DIR, 'latest.epoch')


@lru_cache
def get_s3_filesystem() -> Optional[S3FileSystem]:

    try:
        return S3FileSystem(use_listings_cache=False)
    
    except Exception as e:
        print('Unable to access model checkpoints storage on S3')
        print(e)
        traceback.print_exc()
        return None


def get_checkpoint_filename(epoch: int) -> str:
    return f'weights-epoch-{epoch}'


def get_remote_checkpoint_path(task_id: str, epoch: int) -> str:
    return os.path.join(CHECKPOINTS_BUCKET, task_id, get_checkpoint_filename(epoch))


def get_local_path(task_id: str, epoch: int) -> str:
    return os.path.join(tempfile.gettempdir(), 'gigmate-checkpoints', task_id, get_checkpoint_filename(epoch))


def get_task(task_id: Optional[str] = LATEST_TASK_CHECKPOINT_ID) -> Optional[Task]:
    if task_id is None:
        return None

    return Task.get_task(task_id=task_id, project_name=get_clearml_project_name())


def get_task_artifact(fs: Optional[S3FileSystem], task_id: str, epoch: int) -> str:
    if S3_CHECKPOINTS_STORAGE is True:
        local_path = get_local_path(task_id, epoch)

        if not os.path.exists(local_path):
            if fs is None:
                raise Exception("Unable to download checkpoint: S3Filesystem not specified")

            fs.download(get_remote_checkpoint_path(task_id, epoch), local_path)

        return local_path
        
    else:
        task = cast(Task, get_task(task_id))
        return task.artifacts[get_checkpoint_filename(epoch)].get_local_copy()


def get_latest_model_checkpoint_path(task_id: Optional[str] = LATEST_TASK_CHECKPOINT_ID, epoch: int = EPOCH):
    
    if os.path.exists(LATEST_CHECKPOINT_PATH):
        return LATEST_CHECKPOINT_PATH

    if task_id is None:
        return None
    
    fs = get_s3_filesystem()

    return get_task_artifact(fs, task_id, epoch)
