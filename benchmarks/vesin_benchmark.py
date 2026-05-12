#!/usr/bin/env python
"""Benchmark vesin stateless and Verlet neighbor lists on CPU and CUDA.

The default run measures the Python API end-to-end. CUDA timings synchronize
around each call so asynchronous kernels are accounted for in the wall time.
Results are written to CSV first and plotted from that CSV.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np


FIELDNAMES = [
    "backend",
    "variant",
    "mode",
    "stage",
    "n_atoms",
    "density",
    "cutoff",
    "skin",
    "step_sigma",
    "warmup_steps",
    "steps",
    "repeats",
    "pairs",
    "median_ms_per_step",
    "mean_ms_per_step",
    "stdev_ms_per_step",
    "min_ms_per_step",
]


class BenchmarkCase(NamedTuple):
    backend: str
    variant: str
    mode: str
    algorithm: str


SERIES_STYLES = {
    ("cpu", "cell_list", "stateless"): {
        "label": "CPU cell-list pre-Verlet",
        "color": "#4B5563",
        "marker": "o",
        "linestyle": "--",
    },
    ("cpu", "cell_list", "verlet"): {
        "label": "CPU cell-list post-Verlet",
        "color": "#0072B2",
        "marker": "s",
        "linestyle": "-",
    },
    ("cpu", "simd", "stateless"): {
        "label": "CPU SIMD pre-Verlet",
        "color": "#CC79A7",
        "marker": "P",
        "linestyle": "--",
    },
    ("cpu", "simd", "verlet"): {
        "label": "CPU SIMD post-Verlet",
        "color": "#E69F00",
        "marker": "X",
        "linestyle": "-",
    },
    ("gpu", "cuda", "stateless"): {
        "label": "GPU pre-Verlet",
        "color": "#D55E00",
        "marker": "^",
        "linestyle": "--",
    },
    ("gpu", "cuda", "verlet"): {
        "label": "GPU post-Verlet",
        "color": "#009E73",
        "marker": "D",
        "linestyle": "-",
    },
}


def parse_sizes(value: str) -> list[int]:
    sizes = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        size = int(item)
        if size <= 0:
            raise argparse.ArgumentTypeError("atom counts must be positive")
        sizes.append(size)
    if not sizes:
        raise argparse.ArgumentTypeError("at least one atom count is required")
    return sizes


def parse_backends(value: str) -> list[str]:
    backends = [item.strip().lower() for item in value.split(",") if item.strip()]
    unknown = sorted(set(backends) - {"cpu", "gpu"})
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown backend(s): {', '.join(unknown)}")
    return backends


def generate_system(
    n_atoms: int, density: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    box_size = (n_atoms / density) ** (1.0 / 3.0)
    rng = np.random.default_rng(seed)
    positions = rng.random((n_atoms, 3), dtype=np.float64) * box_size
    box = np.eye(3, dtype=np.float64) * box_size
    return np.ascontiguousarray(positions), np.ascontiguousarray(box)


def generate_displacements(
    n_atoms: int,
    n_steps: int,
    sigma: float,
    seed: int,
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [
        np.ascontiguousarray(rng.normal(0.0, sigma, size=(n_atoms, 3)))
        for _ in range(n_steps)
    ]


def summarize_timings(values: list[float]) -> dict[str, float]:
    return {
        "median_ms_per_step": statistics.median(values),
        "mean_ms_per_step": statistics.fmean(values),
        "stdev_ms_per_step": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min_ms_per_step": min(values),
    }


def _cupy_module():
    try:
        import cupy as cp
    except ImportError:
        return None

    try:
        cp.cuda.Device(0).compute_capability
    except cp.cuda.runtime.CUDARuntimeError:
        return None

    return cp


def _sync_gpu(cp: Any | None) -> None:
    if cp is not None:
        cp.cuda.Stream.null.synchronize()


def _to_backend_arrays(
    backend: str,
    positions: np.ndarray,
    box: np.ndarray,
    cp: Any | None,
) -> tuple[Any, Any]:
    if backend == "gpu":
        return cp.asarray(positions), cp.asarray(box)
    return positions.copy(), box.copy()


def _add_displacement(
    points: Any, displacement: np.ndarray, backend: str, cp: Any | None
) -> Any:
    if backend == "gpu":
        return points + cp.asarray(displacement)
    return np.ascontiguousarray(points + displacement)


def _time_compute(
    nl: Any, points: Any, box: Any, backend: str, cp: Any | None
) -> tuple[float, int]:
    _sync_gpu(cp if backend == "gpu" else None)
    start = time.perf_counter()
    first, _second = nl.compute(points, box, periodic=True, quantities="ij", copy=False)
    _sync_gpu(cp if backend == "gpu" else None)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return elapsed_ms, int(first.shape[0])


def _default_variant(backend: str, algorithm: str) -> str:
    if backend == "gpu":
        return "cuda"
    if algorithm == "auto":
        return "simd"
    return algorithm


def benchmark_cases_for_backend(
    backend: str, args: argparse.Namespace
) -> list[BenchmarkCase]:
    cases = [
        BenchmarkCase(
            backend=backend,
            variant=_default_variant(backend, args.algorithm),
            mode=mode,
            algorithm=args.algorithm,
        )
        for mode in ("stateless", "verlet")
    ]

    if backend == "cpu" and args.include_cpu_simd and args.algorithm != "auto":
        cases.extend(
            [
                BenchmarkCase(
                    backend="cpu",
                    variant="simd",
                    mode=mode,
                    algorithm="auto",
                )
                for mode in ("stateless", "verlet")
            ]
        )

    return cases


def benchmark_one(
    *,
    backend: str,
    mode: str,
    positions: np.ndarray,
    box: np.ndarray,
    displacements: list[np.ndarray],
    warmup_displacements: list[np.ndarray],
    cutoff: float,
    skin: float,
    algorithm: str,
    repeats: int,
    cp: Any | None,
) -> tuple[list[float], int]:
    from vesin import NeighborList

    timings = []
    pairs = 0
    active_skin = 0.0 if mode == "stateless" else skin

    for _ in range(repeats):
        nl = NeighborList(
            cutoff=cutoff,
            full_list=True,
            sorted=False,
            skin=active_skin,
            algorithm=algorithm,
        )
        points, backend_box = _to_backend_arrays(backend, positions, box, cp)

        for displacement in warmup_displacements:
            points = _add_displacement(points, displacement, backend, cp)
            _time_compute(nl, points, backend_box, backend, cp)

        step_timings = []
        for displacement in displacements:
            points = _add_displacement(points, displacement, backend, cp)
            elapsed_ms, pairs = _time_compute(nl, points, backend_box, backend, cp)
            step_timings.append(elapsed_ms)

        timings.append(statistics.fmean(step_timings))

    return timings, pairs


def run_benchmark(args: argparse.Namespace) -> list[dict[str, Any]]:
    cp = _cupy_module() if "gpu" in args.backends else None
    rows: list[dict[str, Any]] = []

    if "gpu" in args.backends and cp is None:
        print("GPU backend skipped: CuPy with CUDA is not available", flush=True)

    for n_atoms in args.sizes:
        positions, box = generate_system(n_atoms, args.density, args.seed + n_atoms)
        displacements = generate_displacements(
            n_atoms,
            args.steps,
            args.step_sigma,
            args.seed + 10_000 + n_atoms,
        )
        warmup_displacements = generate_displacements(
            n_atoms,
            args.warmup_steps,
            args.step_sigma,
            args.seed + 20_000 + n_atoms,
        )

        for backend in args.backends:
            if backend == "gpu" and cp is None:
                continue
            for case in benchmark_cases_for_backend(backend, args):
                timings, pairs = benchmark_one(
                    backend=case.backend,
                    mode=case.mode,
                    positions=positions,
                    box=box,
                    displacements=displacements,
                    warmup_displacements=warmup_displacements,
                    cutoff=args.cutoff,
                    skin=args.skin,
                    algorithm=case.algorithm,
                    repeats=args.repeats,
                    cp=cp,
                )
                row = {
                    "backend": case.backend,
                    "variant": case.variant,
                    "mode": case.mode,
                    "stage": "pre_verlet"
                    if case.mode == "stateless"
                    else "post_verlet",
                    "n_atoms": n_atoms,
                    "density": args.density,
                    "cutoff": args.cutoff,
                    "skin": 0.0 if case.mode == "stateless" else args.skin,
                    "step_sigma": args.step_sigma,
                    "warmup_steps": args.warmup_steps,
                    "steps": args.steps,
                    "repeats": args.repeats,
                    "pairs": pairs,
                    **summarize_timings(timings),
                }
                rows.append(row)
                print(
                    f"{case.backend:>3s} {case.variant:>9s} {case.mode:>9s} "
                    f"{n_atoms:7d} atoms "
                    f"{row['median_ms_per_step']:9.3f} ms/step, {pairs} pairs",
                    flush=True,
                )

    return rows


def write_rows(rows: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def read_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(newline="") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def group_plot_series(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, str, str], tuple[list[int], list[float], list[float]]]:
    grouped: dict[tuple[str, str, str], list[tuple[int, float, float]]] = {}
    for row in rows:
        backend = str(row["backend"])
        mode = str(row["mode"])
        variant = str(row.get("variant") or _default_variant(backend, "cell_list"))
        key = (backend, variant, mode)
        grouped.setdefault(key, []).append(
            (
                int(row["n_atoms"]),
                float(row["median_ms_per_step"]),
                float(row.get("stdev_ms_per_step", 0.0) or 0.0),
            )
        )

    series = {}
    for key, values in grouped.items():
        values.sort(key=lambda item: item[0])
        atoms, medians, stdevs = zip(*values, strict=True)
        series[key] = (list(atoms), list(medians), list(stdevs))
    return series


def plot_results(rows: list[dict[str, Any]], output: Path) -> None:
    import matplotlib.pyplot as plt

    series = group_plot_series(rows)
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "figure.dpi": 150,
            "savefig.dpi": 240,
        }
    )

    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    for key in (
        ("cpu", "cell_list", "stateless"),
        ("cpu", "cell_list", "verlet"),
        ("cpu", "simd", "stateless"),
        ("cpu", "simd", "verlet"),
        ("gpu", "cuda", "stateless"),
        ("gpu", "cuda", "verlet"),
    ):
        if key not in series:
            continue
        atoms, medians, stdevs = series[key]
        style = SERIES_STYLES[key]
        ax.errorbar(
            atoms,
            medians,
            yerr=stdevs,
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=1.8,
            markersize=5.5,
            capsize=3,
            color=style["color"],
            label=style["label"],
        )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Number of atoms")
    ax.set_ylabel("Median time per MD step (ms)")
    ax.set_title("vesin neighbor-list pre/post Verlet scaling")
    ax.grid(True, alpha=0.28, which="both")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix:
        fig.savefig(output, bbox_inches="tight")
        if output.suffix.lower() != ".svg":
            fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    else:
        fig.savefig(output.with_suffix(".png"), bbox_inches="tight")
        fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="measure CPU/GPU pre/post Verlet timings")
    run.add_argument("--output", type=Path, default=Path("vesin_verlet_prepost.csv"))
    run.add_argument("--plot", type=Path, help="also write a plot to this path")
    run.add_argument(
        "--sizes",
        type=parse_sizes,
        default=parse_sizes("1024,2048,4096,8192,16384,32768"),
    )
    run.add_argument(
        "--backends", type=parse_backends, default=parse_backends("cpu,gpu")
    )
    run.add_argument("--density", type=float, default=0.05)
    run.add_argument("--cutoff", type=float, default=5.0)
    run.add_argument("--skin", type=float, default=1.0)
    run.add_argument("--step-sigma", type=float, default=0.02)
    run.add_argument("--warmup-steps", type=int, default=5)
    run.add_argument("--steps", type=int, default=80)
    run.add_argument("--repeats", type=int, default=3)
    run.add_argument("--seed", type=int, default=20260512)
    run.add_argument(
        "--algorithm", choices=["auto", "brute_force", "cell_list"], default="cell_list"
    )
    run.add_argument(
        "--include-cpu-simd",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="measure CPU auto-dispatch as a SIMD cluster-pair variant",
    )
    run.set_defaults(func=run_command)

    plot = subparsers.add_parser("plot", help="plot one or more benchmark CSV files")
    plot.add_argument("inputs", nargs="+", type=Path)
    plot.add_argument("--output", type=Path, required=True)
    plot.set_defaults(func=plot_command)

    return parser


def run_command(args: argparse.Namespace) -> None:
    rows = run_benchmark(args)
    write_rows(rows, args.output)
    print(f"CSV written to {args.output}", flush=True)
    if args.plot is not None:
        plot_results(rows, args.plot)
        print(f"Plot written to {args.plot}", flush=True)


def plot_command(args: argparse.Namespace) -> None:
    rows = read_rows(args.inputs)
    plot_results(rows, args.output)
    print(f"Plot written to {args.output}", flush=True)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
