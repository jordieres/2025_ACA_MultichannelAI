#!/usr/bin/env python3
"""
Automated scaling benchmark runner.

Runs the worker script with 1, 2, 3 and 4 GPUs by splitting companies across workers
(one process per GPU) and collecting:
- Per-worker stdout/stderr
- Per-worker run_summary.json
- nvidia-smi GPU utilization log (1 Hz)
- Master summaries per GPU count
- Global summary and plots (time vs GPUs, speedup vs GPUs)
Plus:
- max/min worker wall time (makespan signal)
- speedup_vs_1gpu + efficiency (speedup/k)
- optional plot: efficiency vs GPUs
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


def split_round_robin(items: List[str], k: int) -> List[List[str]]:
    chunks: List[List[str]] = [[] for _ in range(k)]
    for i, x in enumerate(items):
        chunks[i % k].append(x)
    return chunks


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


def start_gpu_logger(outdir: Path) -> subprocess.Popen:
    outdir.mkdir(parents=True, exist_ok=True)
    gpu_log = outdir / "gpu_usage.csv"
    cmd = [
        "nvidia-smi",
        "--query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total",
        "--format=csv",
        "-l", "1",
    ]
    # stdout file handle must remain open; simplest: open inside Popen
    return subprocess.Popen(cmd, stdout=open(gpu_log, "w"), stderr=subprocess.DEVNULL)


def stop_proc(p: subprocess.Popen) -> None:
    if p.poll() is not None:
        return
    p.terminate()
    try:
        p.wait(timeout=3)
    except subprocess.TimeoutExpired:
        p.kill()


def _extract_worker_wall(summary: Dict[str, Any]) -> Optional[float]:
    """
    Robustly extract worker wall time from worker run_summary.json.
    Returns None if missing.
    """
    if not isinstance(summary, dict):
        return None
    v = summary.get("total_seconds_wall", None)
    if isinstance(v, (int, float)):
        return float(v)
    return None


def run_one_setting(
    worker_script: str,
    paths_csv: str,
    processed_root: str,
    run_dir: Path,
    gpu_ids: List[int],
    companies: List[str],
    plot_company: str,
    cpu_threads_per_worker: int,
) -> Dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)

    # GPU logger for this run
    smi_proc = start_gpu_logger(run_dir)

    chunks = split_round_robin(companies, len(gpu_ids))

    procs: List[Dict[str, Any]] = []
    t0 = time.perf_counter()

    # Use same python interpreter as this script (important for venv/poetry)
    py = sys.executable

    for wi, (gpu, chunk) in enumerate(zip(gpu_ids, chunks)):
        wdir = run_dir / f"worker_{wi}_gpu{gpu}"
        wdir.mkdir(parents=True, exist_ok=True)

        companies_json = wdir / "companies.json"
        companies_json.write_text(json.dumps(chunk, indent=2))

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)

        # Avoid CPU oversubscription across processes
        env["OMP_NUM_THREADS"] = str(cpu_threads_per_worker)
        env["MKL_NUM_THREADS"] = str(cpu_threads_per_worker)

        cmd = [
            py,
            worker_script,
            "--paths_csv", paths_csv,
            "--processed_root", processed_root,
            "--outdir", str(wdir),
            "--companies_json", str(companies_json),
        ]
        if plot_company:
            cmd += ["--plot_company", plot_company]

        stdout_f = open(wdir / "stdout.txt", "w")
        stderr_f = open(wdir / "stderr.txt", "w")

        p = subprocess.Popen(cmd, env=env, stdout=stdout_f, stderr=stderr_f)
        procs.append({
            "proc": p,
            "stdout_f": stdout_f,
            "stderr_f": stderr_f,
            "wdir": wdir,
            "gpu": gpu,
            "n_companies": len(chunk),
        })

    exit_codes: List[int] = []
    for d in procs:
        exit_codes.append(d["proc"].wait())
        d["stdout_f"].close()
        d["stderr_f"].close()

    t1 = time.perf_counter()
    stop_proc(smi_proc)

    # Load worker summaries
    workers = []
    worker_wall_times: List[float] = []
    for d in procs:
        sfile = d["wdir"] / "run_summary.json"
        if sfile.exists():
            s = json.loads(sfile.read_text())
        else:
            s = {"error": "missing run_summary.json"}

        w_wall = _extract_worker_wall(s)
        if w_wall is not None:
            worker_wall_times.append(w_wall)

        workers.append({
            "gpu": d["gpu"],
            "n_companies": d["n_companies"],
            "dir": str(d["wdir"]),
            "summary": s,
        })

    max_worker_wall_time = max(worker_wall_times) if worker_wall_times else None
    min_worker_wall_time = min(worker_wall_times) if worker_wall_times else None

    master = {
        "n_gpus": len(gpu_ids),
        "gpu_ids": gpu_ids,
        "n_companies_total": len(companies),
        "seconds_wall_total": t1 - t0,
        # makespan signal: slowest worker wall time (from worker summaries)
        "max_worker_wall_time": max_worker_wall_time,
        "min_worker_wall_time": min_worker_wall_time,
        "exit_codes": exit_codes,
        "workers": workers,
    }
    write_json(run_dir / "master_summary.json", master)
    return master


def add_speedup_efficiency(masters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Adds:
      - speedup_vs_1gpu = T1/Tk
      - efficiency = speedup/k
    Uses seconds_wall_total as Tk.
    """
    masters_sorted = sorted(masters, key=lambda m: m["n_gpus"])
    if not masters_sorted or masters_sorted[0]["n_gpus"] != 1:
        raise ValueError("gpu_counts must include 1 to compute speedup/efficiency baseline")

    t1 = float(masters_sorted[0]["seconds_wall_total"])

    for m in masters_sorted:
        tk = float(m["seconds_wall_total"])
        k = int(m["n_gpus"])
        speedup = (t1 / tk) if tk > 0 else None
        efficiency = (speedup / k) if (speedup is not None and k > 0) else None
        m["speedup_vs_1gpu"] = speedup
        m["efficiency"] = efficiency

    return masters_sorted


def make_plots(masters: List[Dict[str, Any]], outdir: Path) -> None:
    import matplotlib.pyplot as plt

    masters = sorted(masters, key=lambda m: m["n_gpus"])
    g = [m["n_gpus"] for m in masters]
    t = [m["seconds_wall_total"] for m in masters]

    base = t[0]
    speedup = [base / x for x in t]
    efficiency = [speedup[i] / g[i] for i in range(len(g))]

    outdir.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(g, t, marker="o")
    plt.xlabel("Number of GPUs")
    plt.ylabel("Wall time (s)")
    plt.title("Wall time vs GPUs")
    plt.grid(True)
    plt.savefig(outdir / "time_vs_gpus.png", dpi=200)

    plt.figure()
    plt.plot(g, speedup, marker="o")
    plt.xlabel("Number of GPUs")
    plt.ylabel("Speedup (T1 / Tk)")
    plt.title("Speedup vs GPUs")
    plt.grid(True)
    plt.savefig(outdir / "speedup_vs_gpus.png", dpi=200)

    # Optional efficiency plot (requested)
    plt.figure()
    plt.plot(g, efficiency, marker="o")
    plt.xlabel("Number of GPUs")
    plt.ylabel("Efficiency (Speedup / k)")
    plt.title("Parallel efficiency vs GPUs")
    plt.grid(True)
    plt.savefig(outdir / "efficiency_vs_gpus.png", dpi=200)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--worker_script", required=True, help="Path to study_answers_cli.py")
    p.add_argument("--paths_csv", required=True)
    p.add_argument("--processed_root", required=True)
    p.add_argument("--outdir", required=True)

    p.add_argument("--plot_company", default="")
    p.add_argument("--limit_companies", type=int, default=0)
    p.add_argument("--gpu_counts", default="1,2,3,4", help="e.g. '1,2,3,4'")
    p.add_argument("--physical_gpus", default="0,1,2,3", help="e.g. '0,1,2,3'")
    p.add_argument("--cpu_threads_per_worker", type=int, default=4)

    args = p.parse_args()

    root = Path(args.outdir)
    root.mkdir(parents=True, exist_ok=True)

    # Load companies
    data = pd.read_csv(args.paths_csv)
    companies = data["company"].unique().tolist()
    if args.limit_companies and args.limit_companies > 0:
        companies = companies[: args.limit_companies]

    gpu_counts = [int(x.strip()) for x in args.gpu_counts.split(",") if x.strip()]
    physical = [int(x.strip()) for x in args.physical_gpus.split(",") if x.strip()]

    meta = {
        "worker_script": args.worker_script,
        "paths_csv": args.paths_csv,
        "processed_root": args.processed_root,
        "plot_company": args.plot_company,
        "limit_companies": args.limit_companies,
        "n_companies": len(companies),
        "gpu_counts": gpu_counts,
        "physical_gpus": physical,
        "cpu_threads_per_worker": args.cpu_threads_per_worker,
        "python_executable": sys.executable,
    }
    write_json(root / "bench_meta.json", meta)

    masters: List[Dict[str, Any]] = []

    for c in gpu_counts:
        if c > len(physical):
            raise ValueError(f"Requested {c} GPUs but only {len(physical)} in --physical_gpus")
        gpu_ids = physical[:c]
        run_dir = root / f"gpus_{c}"

        m = run_one_setting(
            worker_script=args.worker_script,
            paths_csv=args.paths_csv,
            processed_root=args.processed_root,
            run_dir=run_dir,
            gpu_ids=gpu_ids,
            companies=companies,
            plot_company=args.plot_company,
            cpu_threads_per_worker=args.cpu_threads_per_worker,
        )
        masters.append(m)

        # Hard fail if any worker failed: prevents "Done" lies
        if any(code != 0 for code in m["exit_codes"]):
            raise RuntimeError(
                f"Workers failed for {c} GPUs. Exit codes: {m['exit_codes']}. "
                f"See logs under: {run_dir}"
            )

    # Add speedup/efficiency fields
    masters = add_speedup_efficiency(masters)

    write_json(root / "bench_results.json", masters)
    make_plots(masters, root)

    print(f"Done. Results in: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())