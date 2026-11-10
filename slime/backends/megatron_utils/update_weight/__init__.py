from __future__ import annotations

from argparse import Namespace
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch


def create_weight_updater(
    args: Namespace,
    model: Sequence[torch.nn.Module],
    weights_getter: Callable[[], Mapping[str, torch.Tensor]],
    *,
    model_name: str,
    quantization_config: dict[str, Any] | None,
    weight_source=None,
):
    """Select and construct the weight updater for the configured transport.

    ``weight_source`` (``metadata`` and ``iter_full()`` of full logical tensors) replaces the
    Megatron-to-HF conversion; the selected native transport and its control flow are unchanged.
    """
    update_weight_mode = args.update_weight_mode
    update_weight_transport = args.update_weight_transport

    if update_weight_mode == "delta":
        # Delta sync is disk-transport only: each engine's /pull_weights applies the published
        # deltas into a host-local checkpoint on every host it spans, and the engines reload
        # via vanilla update_weights_from_disk.
        assert not args.colocate, "--update-weight-mode=delta is not supported with --colocate"
        assert update_weight_transport == "disk", "--update-weight-mode=delta requires --update-weight-transport=disk"
        from .update_weight_from_disk_delta import UpdateWeightFromDiskDelta

        update_weight_cls = UpdateWeightFromDiskDelta
    elif update_weight_transport == "disk":
        from .update_weight_from_disk import UpdateWeightFromDisk

        update_weight_cls = UpdateWeightFromDisk
    elif args.colocate:
        from .update_weight_from_tensor import UpdateWeightFromTensor

        update_weight_cls = UpdateWeightFromTensor
    else:
        assert update_weight_mode == "full"
        assert (
            update_weight_transport == "nccl"
        ), f"unsupported weight sync mode/transport: {update_weight_mode!r}/{update_weight_transport!r}"
        from .update_weight_from_distributed import UpdateWeightFromDistributed

        update_weight_cls = UpdateWeightFromDistributed

    kwargs = {}
    if weight_source is not None:
        if update_weight_cls.__name__ not in ("UpdateWeightFromTensor", "UpdateWeightFromDistributed"):
            raise ValueError(f"a weight source is supported by the tensor and NCCL transports, not {update_weight_cls}")
        kwargs["weight_source"] = weight_source
    updater = update_weight_cls(
        args,
        model,
        weights_getter,
        model_name=model_name,
        quantization_config=quantization_config,
        **kwargs,
    )
    updater.weight_version = getattr(args, "update_weight_start_version", 0)
    return updater
