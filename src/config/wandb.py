from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Mapping, Sequence

import wandb


@dataclass(frozen=True)
class WandbConfig:
    project: str | None = None
    entity: str | None = None
    name: str | None = None
    job_type: str | None = None
    api_key: str | None = None
    config: Mapping[str, Any] | None = None
    tags: Sequence[str] | None = None


def setup_wandb(config: WandbConfig) -> wandb.sdk.wandb_run.Run:
    """Authenticate and enable Ultralytics W&B logging for training.

    Ultralytics registers its W&B callbacks at import time, so this must run
    before the first ``YOLO(...)`` call. ``project_dir`` in ``train_yolo`` is
    only the local save path; W&B project/entity/name come from this config.
    """
    if config.api_key:
        wandb.login(key=config.api_key)

    from ultralytics import settings

    settings.update({"wandb": True})

    project = config.project or os.getenv("WANDB_PROJECT")
    if not project:
        raise ValueError(
            "W&B project is required. Set WANDB_PROJECT or pass WandbConfig.project."
        )

    entity = config.entity or os.getenv("WANDB_ENTITY")
    config_payload = dict(config.config) if config.config is not None else None
    tags = list(config.tags) if config.tags is not None else None

    if wandb.run is not None:
        return wandb.run

    return wandb.init(
        project=project,
        entity=entity,
        name=config.name,
        job_type=config.job_type,
        config=config_payload,
        tags=tags,
    )


def init_wandb(config: WandbConfig) -> wandb.sdk.wandb_run.Run:
    """Deprecated alias for :func:`setup_wandb`."""
    return setup_wandb(config)
