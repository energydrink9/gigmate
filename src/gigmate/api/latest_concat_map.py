from typing import Callable, Optional, TypeVar

from reactivex import Observable, abc, compose
from reactivex.disposable import CompositeDisposable, SingleAssignmentDisposable
from reactivex.internal import synchronized
from reactivex.typing import (
    Mapper,
)
import reactivex.operators as ops

_T = TypeVar("_T")
_T1 = TypeVar("_T1")
_T2 = TypeVar("_T2")


def merge_latest(  # noqa: C901
    max_concurrent: Optional[int] = None
) -> Callable[[Observable[Observable[_T]]], Observable[_T]]:
    def merge(source: Observable[Observable[_T]]) -> Observable[_T]:
        
        def subscribe(
            observer: abc.ObserverBase[_T],
            scheduler: Optional[abc.SchedulerBase] = None,
        ):
            active_count = [0]
            group = CompositeDisposable()
            is_stopped = [False]
            latest_item: Optional[Observable[_T]] = None

            def subscribe(xs: Observable[_T]):
                subscription = SingleAssignmentDisposable()
                group.add(subscription)

                @synchronized(source.lock)
                def on_completed():
                    group.remove(subscription)
                    if latest_item is not None:
                        subscribe(latest_item)
                    else:
                        active_count[0] -= 1
                        if is_stopped[0] and active_count[0] == 0:
                            observer.on_completed()

                on_next = synchronized(source.lock)(observer.on_next)
                on_error = synchronized(source.lock)(observer.on_error)
                subscription.disposable = xs.subscribe(
                    on_next, on_error, on_completed, scheduler=scheduler
                )

            def on_next(inner_source: Observable[_T]) -> None:
                nonlocal latest_item

                assert max_concurrent
                if active_count[0] < max_concurrent:
                    active_count[0] += 1
                    subscribe(inner_source)
                else:
                    latest_item = inner_source

            def on_completed():
                is_stopped[0] = True
                if active_count[0] == 0:
                    observer.on_completed()

            group.add(
                source.subscribe(
                    on_next, observer.on_error, on_completed, scheduler=scheduler
                )
            )
            return group

        return Observable(subscribe)

    return merge


def latest_concat_map(
    project: Mapper[_T1, Observable[_T2]], max_concurrent: Optional[int] = 1
) -> Callable[[Observable[_T1]], Observable[_T2]]:
    """
    Merges the items emitted by an Observable of Observables into a single Observable, 
    ensuring that only the latest inner Observable is subscribed to sequentially, but 
    without canceling any inner Observable once it has begun.

    This operator is similar to `concatMap` and `switchMap`, combining the characteristics 
    of both. Unlike `concatMap`, it takes the latest item like `switchMap`, but unlike 
    `switchMap`, it does not cancel an inner Observable once subscribed.

    Args:
        max_concurrent: The maximum number of inner Observables that can be subscribed 
        to concurrently. This argument is optional, with the default ensuring only 
        sequential execution.

    Returns:
        A function that takes an Observable of Observables and returns a flattened 
        Observable sequence.
    """
    
    return compose(ops.map(project), merge_latest(max_concurrent=max_concurrent))