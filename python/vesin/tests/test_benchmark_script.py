import importlib.util
from argparse import Namespace
from pathlib import Path


def _load_benchmark_module():
    script = Path(__file__).resolve().parents[3] / "benchmarks" / "vesin_benchmark.py"
    spec = importlib.util.spec_from_file_location("vesin_benchmark", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_sizes_accepts_comma_separated_atoms():
    benchmark = _load_benchmark_module()

    assert benchmark.parse_sizes("64, 256,1024") == [64, 256, 1024]


def test_group_plot_series_orders_each_backend_mode_by_size():
    benchmark = _load_benchmark_module()
    rows = [
        {
            "backend": "gpu",
            "mode": "verlet",
            "n_atoms": "256",
            "median_ms_per_step": "0.5",
            "stdev_ms_per_step": "0.1",
        },
        {
            "backend": "gpu",
            "mode": "verlet",
            "n_atoms": "64",
            "median_ms_per_step": "0.2",
            "stdev_ms_per_step": "0.05",
        },
    ]

    series = benchmark.group_plot_series(rows)

    assert series[("gpu", "verlet")] == ([64, 256], [0.2, 0.5], [0.05, 0.1])


def test_default_benchmark_cases_include_cpu_simd_stateless_and_verlet():
    benchmark = _load_benchmark_module()
    args = Namespace(algorithm="cell_list", include_cpu_simd=True)

    cases = benchmark.benchmark_cases_for_backend("cpu", args)

    assert [(case.variant, case.mode, case.algorithm) for case in cases] == [
        ("cell_list", "stateless", "cell_list"),
        ("cell_list", "verlet", "cell_list"),
        ("simd", "stateless", "auto"),
        ("simd", "verlet", "auto"),
    ]


def test_group_plot_series_keeps_cpu_simd_separate_from_cell_list():
    benchmark = _load_benchmark_module()
    rows = [
        {
            "backend": "cpu",
            "variant": "cell_list",
            "mode": "stateless",
            "n_atoms": "512",
            "median_ms_per_step": "2.0",
            "stdev_ms_per_step": "0.2",
        },
        {
            "backend": "cpu",
            "variant": "simd",
            "mode": "verlet",
            "n_atoms": "512",
            "median_ms_per_step": "1.1",
            "stdev_ms_per_step": "0.1",
        },
    ]

    series = benchmark.group_plot_series(rows)

    assert series[("cpu", "cell_list", "stateless")] == (
        [512],
        [2.0],
        [0.2],
    )
    assert series[("cpu", "simd", "verlet")] == ([512], [1.1], [0.1])
