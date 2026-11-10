"""Weight-transfer failures must reach ranks with no local Ray RPC."""

import sys
from types import SimpleNamespace

import pytest
import torch.distributed as dist

from slime.utils.weight_sync import iter_source_chunks, wait_weight_updates


@pytest.mark.unit
@pytest.mark.parametrize("sender", [False, True])
def test_wait_weight_updates_refuses_sender_failure_on_every_rank(monkeypatch, sender):
    def get(refs):
        if sender:
            raise ValueError("receiver rejected applied bytes")

    monkeypatch.setitem(sys.modules, "ray", SimpleNamespace(get=get))
    monkeypatch.setattr(dist, "get_rank", lambda: 1 if sender else 0)
    monkeypatch.setattr(dist, "get_world_size", lambda group: 2)

    def gather(errors, error, group):
        assert (error is not None) == sender
        errors[:] = [None, "rank 1: receiver rejected applied bytes"]

    monkeypatch.setattr(dist, "all_gather_object", gather)
    with pytest.raises(RuntimeError, match="generation stays paused.*receiver rejected"):
        wait_weight_updates(["rpc"] if sender else [], "gloo")


@pytest.mark.unit
def test_source_chunking_preserves_both_passes_and_mixed_dtypes():
    import torch

    weights = [("a", torch.ones(3, dtype=torch.bfloat16)), ("b", torch.ones(2, dtype=torch.float32))]
    source = SimpleNamespace(iter_full=lambda: iter(weights * 2))
    chunks = list(iter_source_chunks(source, 10))
    assert [[name for name, tensor in chunk] for chunk in chunks] == [["a"], ["b"], ["a"], ["b"]]
