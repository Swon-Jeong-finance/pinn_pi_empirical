from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import zipfile
import tempfile
import shutil
import re
import argparse
import yaml

TARGET_STRATEGIES = ["pipinn", "myopic", "pipinn_zero"]

def max_drawdown_from_wealth(wealth: pd.Series) -> float:
    running_max = wealth.cummax()
    drawdown = wealth / running_max - 1.0
    return float(drawdown.min())

def compute_metrics_from_returns(ret: pd.Series, turnover: pd.Series = None, risky_weight: pd.Series = None, gamma: float = 5.0) -> dict:
    ret = pd.to_numeric(ret, errors="coerce").dropna()
    if len(ret) == 0:
        return {
            "months": 0, "ann_ret": np.nan, "ann_vol": np.nan, "sharpe": np.nan,
            "cer_ann": np.nan, "avg_turnover": np.nan, "avg_risky_weight": np.nan,
            "max_drawdown": np.nan
        }
    wealth = (1.0 + ret).cumprod()
    ann_ret = wealth.iloc[-1] ** (12.0 / len(ret)) - 1.0
    ann_vol = ret.std(ddof=1) * np.sqrt(12.0) if len(ret) > 1 else np.nan
    sharpe = (ret.mean() / ret.std(ddof=1) * np.sqrt(12.0)) if len(ret) > 1 and ret.std(ddof=1) > 0 else np.nan
    cer_ann = 12.0 * (ret.mean() - 0.5 * gamma * ret.var(ddof=1)) if len(ret) > 1 else 12.0 * ret.mean()
    avg_turnover = float(pd.to_numeric(turnover, errors="coerce").mean()) if turnover is not None else np.nan
    avg_risky_weight = float(pd.to_numeric(risky_weight, errors="coerce").mean()) if risky_weight is not None else np.nan
    mdd = max_drawdown_from_wealth(wealth)
    return {
        "months": int(len(ret)),
        "ann_ret": float(ann_ret),
        "ann_vol": float(ann_vol) if pd.notna(ann_vol) else np.nan,
        "sharpe": float(sharpe) if pd.notna(sharpe) else np.nan,
        "cer_ann": float(cer_ann),
        "avg_turnover": avg_turnover,
        "avg_risky_weight": avg_risky_weight,
        "max_drawdown": float(mdd),
    }

def has_target_csvs(root: Path) -> bool:
    return any(re.match(r"[^_]+_rank\d+_monthly_paths(_oos)?\.csv$", p.name) for p in root.rglob("*.csv"))

def extract_zip_to_temp(zip_path: Path):
    cleanup_dir = Path(tempfile.mkdtemp(prefix="ff25_zip_extract_"))
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(cleanup_dir)
    return cleanup_dir, cleanup_dir

def prepare_input_root(input_path: Path):
    cleanup_dir = None

    if input_path.is_file() and input_path.suffix.lower() == ".zip":
        return extract_zip_to_temp(input_path)

    if input_path.is_dir():
        if has_target_csvs(input_path):
            return input_path, cleanup_dir

        zip_candidates = sorted(input_path.rglob("*.zip"))
        for z in zip_candidates:
            temp_root, cleanup_dir = extract_zip_to_temp(z)
            if has_target_csvs(temp_root):
                return temp_root, cleanup_dir
            shutil.rmtree(temp_root, ignore_errors=True)

        raise FileNotFoundError(
            f"No target CSVs found in directory and no usable zip found inside: {input_path}"
        )

    raise FileNotFoundError(f"Input must be a directory or zip file: {input_path}")

def build_file_map(root: Path):
    pattern = re.compile(r"(?P<label>[^_]+)_rank(?P<rank>\d+)_(?P<tail>.+)\.csv$")
    file_map = {}
    target_labels = set()
    for f in root.rglob("*.csv"):
        m = pattern.match(f.name)
        if not m:
            continue
        label = m.group("label")
        rank = int(m.group("rank"))
        tail = m.group("tail")
        target_labels.add(label)
        file_map.setdefault(rank, {})
        file_map[rank][tail] = f
    return file_map, sorted(target_labels)

def find_required_file(file_map, rank: int, tail: str) -> Path:
    try:
        return file_map[rank][tail]
    except KeyError:
        raise FileNotFoundError(f"Missing file for rank {rank}: ff25_rank{rank}_{tail}.csv")

def load_monthly(file_map, rank: int, oos: bool = False) -> pd.DataFrame:
    tail = "monthly_paths_oos" if oos else "monthly_paths"
    f = find_required_file(file_map, rank, tail)
    df = pd.read_csv(f)
    df["return_date"] = pd.to_datetime(df["return_date"])
    df = df[df["strategy"].isin(TARGET_STRATEGIES)].copy()
    return df.sort_values(["strategy", "return_date"]).reset_index(drop=True)

def _summary_tail_candidates(summary_kind: str) -> list[str]:
    if summary_kind == "all":
        return ["comparison_cross_modes_all_costs_summary"]
    if summary_kind == "zero":
        return ["comparison_cross_modes_zero_cost_summary"]
    # auto: all 우선, 없으면 zero fallback
    return ["comparison_cross_modes_all_costs_summary", "comparison_cross_modes_zero_cost_summary"]

def load_summary(file_map, rank: int, oos: bool = False, summary_kind: str = "all") -> pd.DataFrame:
    f = None
    for base in _summary_tail_candidates(summary_kind):
        tail = f"{base}_oos" if oos else base
        try:
            f = find_required_file(file_map, rank, tail)
            break
        except FileNotFoundError:
            continue
    if f is None:
        raise FileNotFoundError(f"Missing summary file for rank={rank}, oos={oos}, kind={summary_kind}")
    df = pd.read_csv(f)
    df = df[df["strategy"].isin(TARGET_STRATEGIES)].copy()
    wanted = [
        "strategy", "months", "ann_ret", "ann_vol", "sharpe", "cer_ann",
        "avg_turnover", "avg_risky_weight", "max_drawdown"
    ]
    return df[wanted].sort_values("strategy").reset_index(drop=True)

def build_combined_summary(file_map, rank: int, gamma: float = 5.0) -> pd.DataFrame:
    is_df = load_monthly(file_map, rank, oos=False)
    oos_df = load_monthly(file_map, rank, oos=True)
    combined = pd.concat([is_df, oos_df], axis=0, ignore_index=True)
    combined = combined.sort_values(["strategy", "return_date"]).reset_index(drop=True)

    rows = []
    for strategy, g in combined.groupby("strategy", sort=False):
        metrics = compute_metrics_from_returns(
            ret=g["net_return"],
            turnover=g["turnover"],
            risky_weight=g["risky_weight"],
            gamma=gamma,
        )
        metrics["strategy"] = strategy
        rows.append(metrics)

    cols = ["strategy", "months", "ann_ret", "ann_vol", "sharpe", "cer_ann", "avg_turnover", "avg_risky_weight", "max_drawdown"]
    return pd.DataFrame(rows)[cols].sort_values("strategy").reset_index(drop=True)

def build_monthly_panel(file_map, rank: int) -> pd.DataFrame:
    is_df = load_monthly(file_map, rank, oos=False).copy()
    is_df["sample"] = "IS"

    oos_df = load_monthly(file_map, rank, oos=True).copy()
    oos_df["sample"] = "OOS"

    all_df = pd.concat([is_df, oos_df], axis=0, ignore_index=True)
    all_df = all_df.sort_values(["strategy", "return_date"]).reset_index(drop=True)

    frames = []
    for df in [is_df, oos_df]:
        tmp = df.copy()
        tmp["wealth"] = tmp.groupby("strategy")["net_return"].transform(lambda x: (1 + x).cumprod())
        frames.append(tmp)

    full_panel = all_df.copy()
    full_panel["sample"] = "IS+OOS"
    full_panel["wealth"] = full_panel.groupby("strategy")["net_return"].transform(lambda x: (1 + x).cumprod())
    frames.append(full_panel)

    panel = pd.concat(frames, axis=0, ignore_index=True)
    panel = panel[["strategy", "sample", "return_date", "net_return", "gross_return", "turnover", "risky_weight", "wealth"]]
    return panel.sort_values(["sample", "strategy", "return_date"]).reset_index(drop=True)

def save_plot(panel: pd.DataFrame, rank: int, outdir: Path, target_label: str):
    for sample in ["IS", "OOS", "IS+OOS"]:
        fig, ax = plt.subplots(figsize=(10, 5))
        sub = panel[panel["sample"] == sample].copy()
        for strategy in TARGET_STRATEGIES:
            g = sub[sub["strategy"] == strategy].sort_values("return_date")
            if len(g) == 0:
                continue
            ax.plot(g["return_date"], g["wealth"], label=strategy)
        ax.set_title(f"{target_label.upper()} Rank {rank} - Cumulative Wealth ({sample})")
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative wealth")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        suffix = sample.lower().replace("+", "plus")
        fig.savefig(outdir / f"{target_label}_rank{rank}_cumwealth_{suffix}.png", dpi=160)
        plt.close(fig)

def merge_three_summaries(file_map, rank: int, gamma: float = 5.0, summary_kind: str = "all") -> pd.DataFrame:
    is_sum = load_summary(file_map, rank, oos=False, summary_kind=summary_kind).rename(columns=lambda c: f"IS_{c}" if c != "strategy" else c)
    oos_sum = load_summary(file_map, rank, oos=True, summary_kind=summary_kind).rename(columns=lambda c: f"OOS_{c}" if c != "strategy" else c)
    full_sum = build_combined_summary(file_map, rank, gamma=gamma).rename(columns=lambda c: f"IS_OOS_{c}" if c != "strategy" else c)
    merged = is_sum.merge(oos_sum, on="strategy", how="outer").merge(full_sum, on="strategy", how="outer").sort_values("strategy").reset_index(drop=True)
    merged.insert(0, "rank", rank)
    return merged

def infer_target_label(experiment_root: Path) -> str:
    # ex) experiments/ff25/pls_tau1_fixed -> ff25
    # ex) experiments/ff6_pls_tau1_declining -> ff6
    parts = experiment_root.parts
    if "experiments" in parts:
        idx = parts.index("experiments")
        if idx + 1 < len(parts):
            cand = parts[idx + 1]
            if re.fullmatch(r"ff\d+", cand):
                return cand
            m = re.match(r"(ff\d+)", cand)
            if m:
                return m.group(1)
    m = re.match(r"(ff\d+)", experiment_root.name)
    if m:
        return m.group(1)
    return "asset"

def _find_single_file(base: Path, pattern: str) -> Path:
    hits = sorted(base.rglob(pattern))
    if not hits:
        raise FileNotFoundError(f"Missing required file pattern '{pattern}' under {base}")
    return hits[0]

def _selection_file_from_model(selection_root: Path, model: dict, filename: str) -> Path:
    unit_id = model["selection_unit_id"]
    protocol = model.get("selection_protocol_name", "rolling240m_annual")
    stage2_model_label = model["stage2_model_label"]
    unit_dir = selection_root / "stage2_protocol_covariance_eval" / unit_id
    if not unit_dir.exists():
        raise FileNotFoundError(f"Selection unit directory not found: {unit_dir}")
    val_dirs = sorted(unit_dir.glob("valholdout*"))
    if not val_dirs:
        raise FileNotFoundError(f"No valholdout* directory under {unit_dir}")
    # protocol명이 포함된 케이스가 생길 수 있어 우선 필터링
    preferred = [v for v in val_dirs if protocol in v.as_posix()]
    val_dir = preferred[0] if preferred else val_dirs[0]
    return val_dir / stage2_model_label / "outputs" / filename

def collect_rank_sweep_inputs_to_temp(experiment_root: Path, target_label: str, summary_kind: str = "all"):
    selected_spec = experiment_root / "selected_spec.yaml"
    selection_root = experiment_root / "selection"
    if not selected_spec.exists():
        raise FileNotFoundError(f"selected_spec.yaml not found: {selected_spec}")
    if not selection_root.exists():
        raise FileNotFoundError(f"selection directory not found: {selection_root}")

    with selected_spec.open("r", encoding="utf-8") as f:
        spec = yaml.safe_load(f)
    selected_models = spec.get("selected_models", [])
    if not isinstance(selected_models, list) or len(selected_models) == 0:
        raise ValueError(f"No selected_models found in {selected_spec}")

    tmp_root = Path(tempfile.mkdtemp(prefix="rank_sweep_collect_"))
    for i, model in enumerate(selected_models, start=1):
        rank_dir = experiment_root / f"rank_{i:03d}" / "outputs"
        if not rank_dir.exists():
            raise FileNotFoundError(f"Missing OOS rank directory: {rank_dir}")
        oos_monthly_src = _find_single_file(rank_dir, "monthly_paths.csv")
        is_monthly_src = _selection_file_from_model(selection_root, model, "monthly/monthly_paths.csv")

        chosen_tail = None
        oos_summary_src = None
        is_summary_src = None
        for summary_tail in _summary_tail_candidates(summary_kind):
            summary_name = f"{summary_tail}.csv"
            try:
                cand_oos = _find_single_file(rank_dir, summary_name)
                cand_is = _selection_file_from_model(selection_root, model, summary_name)
                if cand_is.exists():
                    chosen_tail = summary_tail
                    oos_summary_src = cand_oos
                    is_summary_src = cand_is
                    break
            except FileNotFoundError:
                continue

        if chosen_tail is None or oos_summary_src is None or is_summary_src is None:
            raise FileNotFoundError(f"Missing summary files for rank {i} with summary_kind={summary_kind}")
        if not is_summary_src.exists() or not is_monthly_src.exists():
            raise FileNotFoundError(f"Missing IS selection outputs for rank {i}: {is_summary_src}, {is_monthly_src}")

        shutil.copy2(is_summary_src, tmp_root / f"{target_label}_rank{i}_{chosen_tail}.csv")
        shutil.copy2(is_monthly_src, tmp_root / f"{target_label}_rank{i}_monthly_paths.csv")
        shutil.copy2(oos_summary_src, tmp_root / f"{target_label}_rank{i}_{chosen_tail}_oos.csv")
        shutil.copy2(oos_monthly_src, tmp_root / f"{target_label}_rank{i}_monthly_paths_oos.csv")

    return tmp_root

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str, required=True, help="Zip file, directory with CSVs, or directory containing a zip")
    parser.add_argument("--out-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--gamma", type=float, default=5.0, help="Risk aversion used in CER calculation")
    parser.add_argument("--target-label", type=str, default="auto", help="Output filename prefix (e.g., ff25). 'auto' infers from path")
    parser.add_argument("--summary-kind", type=str, choices=["all", "zero", "auto"], default="all", help="Summary preference: all(default), zero, or auto(all->zero fallback)")
    args = parser.parse_args()

    input_path = Path(args.input_path)
    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    cleanup_dir = None
    collected_dir = None
    # rank-sweep 실험 루트 직접 입력 시, 표준 파일명으로 임시 수집 후 기존 파이프라인에 연결
    if input_path.is_dir() and (input_path / "selected_spec.yaml").exists() and (input_path / "selection").exists():
        target_label = infer_target_label(input_path) if args.target_label == "auto" else args.target_label
        collected_dir = collect_rank_sweep_inputs_to_temp(input_path, target_label=target_label, summary_kind=args.summary_kind)
        root = collected_dir
    else:
        root, cleanup_dir = prepare_input_root(input_path)
        if args.target_label == "auto":
            # 기존 ff25_rank*.csv 번들 입력과의 호환
            _, labels = build_file_map(root)
            target_label = labels[0] if len(labels) == 1 else infer_target_label(input_path)
        else:
            target_label = args.target_label
    try:
        file_map, _ = build_file_map(root)
        ranks = sorted(file_map.keys())
        if not ranks:
            raise FileNotFoundError("No <target>_rank*_*.csv files were found in the input path.")

        combined_summary_frames = []
        combined_panel_frames = []

        for rank in ranks:
            panel = build_monthly_panel(file_map, rank)
            panel.insert(0, "rank", rank)
            panel.to_csv(outdir / f"{target_label}_rank{rank}_monthly_panel_with_wealth.csv", index=False)

            summary = merge_three_summaries(file_map, rank, gamma=args.gamma, summary_kind=args.summary_kind)
            summary.to_csv(outdir / f"{target_label}_rank{rank}_summary_IS_OOS_full.csv", index=False)

            save_plot(panel.drop(columns=["rank"]), rank, outdir, target_label=target_label)

            combined_summary_frames.append(summary)
            combined_panel_frames.append(panel)

        all_summary = pd.concat(combined_summary_frames, axis=0, ignore_index=True)
        all_summary = all_summary.sort_values(["rank", "strategy"]).reset_index(drop=True)
        all_summary.to_csv(outdir / f"{target_label}_all_ranks_summary_IS_OOS_full.csv", index=False)

        all_panel = pd.concat(combined_panel_frames, axis=0, ignore_index=True)
        all_panel = all_panel.sort_values(["rank", "sample", "strategy", "return_date"]).reset_index(drop=True)
        all_panel.to_csv(outdir / f"{target_label}_all_ranks_monthly_panel_with_wealth.csv", index=False)

        print(f"Done. Saved outputs to: {outdir}")
        print(f"Ranks found: {ranks}")
    finally:
        if collected_dir is not None and collected_dir.exists():
            shutil.rmtree(collected_dir, ignore_errors=True)
        if cleanup_dir is not None and cleanup_dir.exists():
            shutil.rmtree(cleanup_dir, ignore_errors=True)

if __name__ == "__main__":
    main()
