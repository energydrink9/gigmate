import os
from typing import List, Optional, Tuple, Union
import coiled
import dask.config
from dask.distributed import Client, LocalCluster

NUM_WORKERS = [4, 50]


def get_client(
    run_locally: bool,
    mount_bucket: Optional[str] = None,
    n_workers: Union[int, List[int]] = NUM_WORKERS,
    worker_vm_types: List[str] = ["c4.large"],
    scheduler_vm_types: List[str] = ['c4.large'],
) -> Union[Client, Tuple[Client, str]]:

    dask.config.set({'distributed.scheduler.allowed-failures': 12})

    if run_locally is True:
        cluster = LocalCluster()

    else:
        cluster = coiled.Cluster(
            n_workers=n_workers,
            package_sync_conda_extras=['ffmpeg'],
            mount_bucket=mount_bucket,
            idle_timeout="5 minutes",
            worker_vm_types=worker_vm_types,
            scheduler_vm_types=scheduler_vm_types,
        )

    client = cluster.get_client()

    if mount_bucket is not None:
        mount_point = '../dataset' if run_locally else os.path.join('/mount', mount_bucket.replace('s3://', ''))
        return client, mount_point
    
    else:
        return client