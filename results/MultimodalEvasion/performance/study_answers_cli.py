#!/usr/bin/env python3
"""
Run the QA analysis pipeline for a subset of companies and write a per-worker summary.

This script is designed to be invoked by a launcher that sets CUDA_VISIBLE_DEVICES
to pin the process to a single GPU.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import spacy
from spacy.cli import download as spacy_download
from bertopic import BERTopic
from keybert import KeyBERT
from umap import UMAP

from multimodal_fin.analysis_qa_effects import (
    PipelineConfig,
    MultiCompanyRunner,
    CompanyPipeline,
    AnswerPlotter,
    TopicModeler,
    KeywordExtractor,
    TopicLabeler,
    EmotionFeatureBuilder,
    StatsTester,
    TranscriptQALoader,
    TextPreprocessor,
    EmotionAggregator,
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_spacy_model(model_name: str = "en_core_web_sm"):
    try:
        return spacy.load(model_name)
    except OSError:
        spacy_download(model_name)
        return spacy.load(model_name)


def safe_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths_csv", required=True, type=str)
    parser.add_argument("--processed_root", required=True, type=str)
    parser.add_argument("--outdir", required=True, type=str)

    parser.add_argument("--plot_company", default="", type=str)
    parser.add_argument("--limit_companies", default=0, type=int)
    parser.add_argument(
        "--companies_json",
        default="",
        type=str,
        help="Optional path to a JSON file containing a list of company tickers to process.",
    )
    parser.add_argument("--spacy_model", default="en_core_web_sm", type=str)

    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Basic run metadata
    run_meta: Dict[str, Any] = {
        "timestamp_utc": utc_now_iso(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cwd": str(Path.cwd()),
        "env": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", ""),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS", ""),
        },
        "args": vars(args),
    }
    safe_write_json(outdir / "run_meta.json", run_meta)

    # Load companies
    t0 = time.perf_counter()
    data = pd.read_csv(args.paths_csv)
    all_companies: List[str] = data["company"].unique().tolist()

    if args.companies_json:
        companies = json.loads(Path(args.companies_json).read_text())
    else:
        companies = all_companies

    if args.limit_companies and args.limit_companies > 0:
        companies = companies[: args.limit_companies]

    stage_times: Dict[str, float] = {}
    stage_t0 = time.perf_counter()

    # Build config + pipeline
    cfg = PipelineConfig(processed_root=args.processed_root)
    stage_times["config_init"] = time.perf_counter() - stage_t0

    stage_t0 = time.perf_counter()
    nlp = load_spacy_model(args.spacy_model)
    stage_times["spacy_load"] = time.perf_counter() - stage_t0

    stage_t0 = time.perf_counter()
    topic_model = BERTopic(
        language="english",
        min_topic_size=3,
        embedding_model="multi-qa-distilbert-cos-v1",
        verbose=False,
        umap_model=UMAP(random_state=42),
    )
    qa_kw_model = KeyBERT(model="all-MiniLM-L12-v2")
    topic_kw_model = KeyBERT()
    stage_times["models_init"] = time.perf_counter() - stage_t0

    stage_t0 = time.perf_counter()
    emo_agg = EmotionAggregator()
    loader = TranscriptQALoader(cfg, emo_agg)
    preproc = TextPreprocessor(nlp)
    topic_modeler = TopicModeler(preproc, topic_model)
    kw_extractor = KeywordExtractor(qa_kw_model)
    topic_labeler = TopicLabeler(topic_kw_model)
    feature_builder = EmotionFeatureBuilder(cfg)
    stats = StatsTester(cfg)
    plotter = AnswerPlotter()

    pipeline = CompanyPipeline(
        config=cfg,
        loader=loader,
        topic_modeler=topic_modeler,
        kw_extractor=kw_extractor,
        topic_labeler=topic_labeler,
        feature_builder=feature_builder,
        stats=stats,
        plotter=plotter,
    )
    stage_times["pipeline_init"] = time.perf_counter() - stage_t0

    # Run
    stage_t0 = time.perf_counter()
    runner = MultiCompanyRunner(pipeline, cfg, use_tqdm=False)

    df_all = runner.run(companies, plot_company=args.plot_company or None)
    stage_times["runner_run_total"] = time.perf_counter() - stage_t0

    # Save outputs
    stage_t0 = time.perf_counter()
    # Minimal artifacts: aggregated CSV
    df_path = outdir / "df_all.csv"
    df_all.to_csv(df_path, index=False)
    stage_times["save_outputs"] = time.perf_counter() - stage_t0

    total_seconds = time.perf_counter() - t0

    bench_summary = {
        "timestamp_utc": utc_now_iso(),
        "hostname": socket.gethostname(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "n_companies": len(companies),
        "n_rows_df_all": int(df_all.shape[0]),
        "total_seconds_wall": total_seconds,
        "stage_times": stage_times,
    }
    safe_write_json(outdir / "run_summary.json", bench_summary)

    print("\n=== BENCH SUMMARY ===")
    print(f"companies: {bench_summary['n_companies']}")
    print(f"total_seconds: {bench_summary['total_seconds_wall']:.2f}")
    print("Top stages:")
    for k, v in sorted(stage_times.items(), key=lambda kv: kv[1], reverse=True)[:10]:
        print(f"  {k:>20}: {v:.2f}s")
    print(f"\nOutputs in: {outdir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())