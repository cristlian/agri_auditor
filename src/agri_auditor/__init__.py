"""agri_auditor package."""

from .features import FeatureEngine
from .ingestion import CameraModel, LogLoader
from .config import RuntimeConfig, load_runtime_config
from .intelligence import (
    ALL_CAMERAS,
    Event,
    EventCandidate,
    EventDetector,
    GeminiAnalyst,
    GeminiAnalysisResult,
    GeminiConfigError,
    IntelligenceOrchestrator,
)

__all__ = [
    "ALL_CAMERAS",
    "CameraModel",
    "Event",
    "EventCandidate",
    "EventDetector",
    "FeatureEngine",
    "GeminiAnalyst",
    "GeminiAnalysisResult",
    "GeminiConfigError",
    "IntelligenceOrchestrator",
    "LogLoader",
    "ReportBuilder",
    "RuntimeConfig",
    "load_runtime_config",
]


def __getattr__(name: str) -> object:
    if name == "ReportBuilder":
        try:
            from .reporting import ReportBuilder as _ReportBuilder
        except ImportError as exc:
            raise ImportError(
                "ReportBuilder requires report dependencies. "
                "Install with: pip install agri-auditor[report]"
            ) from exc
        return _ReportBuilder
    raise AttributeError(name)
