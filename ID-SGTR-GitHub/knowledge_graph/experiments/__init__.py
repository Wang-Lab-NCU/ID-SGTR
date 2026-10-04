"""Reusable components for the controlled ID-SGTR experiments."""

from .evidence import EvidenceAssembler, EvidenceItem, EvidenceVariant
from .metrics import evaluate_predictions, paired_bootstrap
from .telemetry import ContextTrackedChatModel, QueryTelemetry, TrackedChatModel, bind_telemetry, record_candidates, record_evidence, record_runtime_execution, record_runtime_step

__all__ = [
    "EvidenceAssembler",
    "EvidenceItem",
    "EvidenceVariant",
    "QueryTelemetry",
    "TrackedChatModel",
    "ContextTrackedChatModel",
    "bind_telemetry",
    "record_evidence",
    "record_candidates",
    "record_runtime_step",
    "record_runtime_execution",
    "evaluate_predictions",
    "paired_bootstrap",
]
