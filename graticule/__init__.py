from __future__ import annotations

from .cartouche import RunConfig, load_sheet, write_sheet
from .chart import Trainer
from .gazetteer import PanelBatch, SyntheticPanels, collate_panels, synthesize_panel
from .legend import CounterfactualAtlas, MixtureHead, discover_subgroups

__version__ = "0.1.0"

__all__ = [
    "RunConfig",
    "load_sheet",
    "write_sheet",
    "PanelBatch",
    "SyntheticPanels",
    "collate_panels",
    "synthesize_panel",
    "CounterfactualAtlas",
    "MixtureHead",
    "discover_subgroups",
    "Trainer",
]
