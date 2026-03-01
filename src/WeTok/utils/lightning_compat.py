from __future__ import annotations

import torch

try:
    import lightning as L  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    class _LightningModule(torch.nn.Module):
        @property
        def device(self) -> torch.device:
            try:
                return next(self.parameters()).device
            except StopIteration:
                return torch.device("cpu")

    class _LightningDataModule(object):
        pass

    class _FallbackLightning(object):
        LightningModule = _LightningModule
        LightningDataModule = _LightningDataModule

    L = _FallbackLightning()

