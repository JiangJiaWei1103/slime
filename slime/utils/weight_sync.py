"""Cache refresh policy shared by the weight transports."""


def should_flush_cache(flush_cache_interval: int, weight_version: int, initial_weight_version: int = 0) -> bool:
    """Refresh at startup and every N subsequent syncs; nonpositive preserves KV.

    The first publication flushes even when resuming with a nonzero version.
    The subsequent schedule keeps its phase relative to version 1.
    A full refresh aborts requests; other updates pause them in place.
    """
    return weight_version == initial_weight_version + 1 or (
        flush_cache_interval > 0 and (weight_version - 1) % flush_cache_interval == 0
    )


def iter_source_chunks(weight_source, buffer_size: int):
    """Bucket a weight source's full logical tensors by ``--update-weight-buffer-size`` bytes.

    Every trainer rank must iterate: the source's stream gathers TP shards and EP-held experts and
    broadcasts pipeline stages collectively. Names, dtypes and shapes are the source's own; no
    framework conversion runs on this path.
    """
    bucket, size = [], 0
    for name, tensor in weight_source.iter_full():
        nbytes = tensor.numel() * tensor.element_size()
        if bucket and size + nbytes > buffer_size:
            yield bucket
            bucket, size = [], 0
        bucket.append((name, tensor))
        size += nbytes
    if bucket:
        yield bucket


def rank0_then_share(fn, group):
    """Run ``fn`` on global rank 0 and raise its failure on every rank of ``group``.

    A refused weight-update round must stop every trainer rank at the same point instead of leaving
    the other ranks waiting in the next barrier.
    """
    import torch.distributed as dist

    status = [None]
    if dist.get_rank() == 0:
        try:
            fn()
        except Exception as error:  # noqa: BLE001 -- every rank re-raises the same message below
            status[0] = f"{type(error).__name__}: {error}"
    dist.broadcast_object_list(status, src=0, group=group)
    if status[0] is not None:
        raise RuntimeError(f"weight update round failed on rank 0; generation stays paused: {status[0]}")


def begin_engine_round(engines) -> None:
    """Open the native weight-update session on every engine that receives this round."""
    import ray

    ray.get([engine.begin_weight_update.remote() for engine in engines])


def finish_engine_round(engines, weight_version) -> None:
    """Close the sessions, then publish the version; a refused close leaves generation paused."""
    import ray

    ray.get([engine.end_weight_update.remote() for engine in engines])
    ray.get([engine.update_weight_version.remote(str(weight_version)) for engine in engines])
