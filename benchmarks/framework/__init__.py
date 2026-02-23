"""Benchmark framework – core classes for config-driven modality benchmarks."""

from benchmarks.framework.benchmark_config import BenchmarkConfig, load_config
from benchmarks.framework.base_benchmark import BaseBenchmark
from benchmarks.framework.metrics import compute_psnr, compute_ssim, compute_sam, MetricSet
from benchmarks.framework.data_source import DataSource
from benchmarks.framework.source_attribution import SourceAttribution
from benchmarks.framework.mismatch_engine import MismatchEngine
from benchmarks.framework.report_writer import ReportWriter, RunBundle
from benchmarks.framework.expanded_config import (
    ExpandedBenchmarkConfig,
    CaseInstance,
    SystemVariant,
    load_expanded_config,
    load_all_expanded_configs,
)

__all__ = [
    "BenchmarkConfig",
    "load_config",
    "BaseBenchmark",
    "compute_psnr",
    "compute_ssim",
    "compute_sam",
    "MetricSet",
    "DataSource",
    "SourceAttribution",
    "MismatchEngine",
    "ReportWriter",
    "RunBundle",
    # Expanded config system
    "ExpandedBenchmarkConfig",
    "CaseInstance",
    "SystemVariant",
    "load_expanded_config",
    "load_all_expanded_configs",
]
