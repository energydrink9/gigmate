import os
from typing import Optional, Tuple, Union
import coiled
from dask.distributed import Client, LocalCluster

NUM_WORKERS = 100


def get_client(run_locally: bool, mount_bucket: Optional[str] = None) -> Union[Client, Tuple[Client, str]]:

    if run_locally is True:
        cluster = LocalCluster()

    else:
        cluster = coiled.Cluster(
            n_workers=NUM_WORKERS,
            package_sync_conda_extras=['ffmpeg'],
            mount_bucket=mount_bucket,
            idle_timeout="5 minutes",
            worker_vm_types=["t3.medium"],
            scheduler_vm_types=['c4.large']
        )

    client = cluster.get_client()

    if mount_bucket is not None:
        mount_point = '../dataset' if run_locally else os.path.join('/mount', mount_bucket.replace('s3://', ''))
        return client, mount_point
    
    else:
        return client