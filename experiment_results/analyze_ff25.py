
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import zipfile
import tempfile
import shutil
import re
import argparse

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
    return any(root.rglob("ff25_rank*_monthly_paths.csv"))

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
    pattern = re.compile(r"ff25_rank(\d+)_(.+)\.csv$")
    file_map = {}
    for f in root.rglob("*.csv"):
        m = pattern.match(f.name)
        if not m:
            continue
        rank = int(m.group(1))
        tail = m.group(2)
        file_map.setdefault(rank, {})
        file_map[rank][tail] = f
    return file_map

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

def load_summary(file_map, rank: int, oos: bool = False) -> pd.DataFrame:
    tail = "comparison_cross_modes_zero_cost_summary_oos" if oos else "comparison_cross_modes_zero_cost_summary"
    f = find_required_file(file_map, rank, tail)
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

def save_plot(panel: pd.DataFrame, rank: int, outdir: Path):
    for sample in ["IS", "OOS", "IS+OOS"]:
        fig, ax = plt.subplots(figsize=(10, 5))
        sub = panel[panel["sample"] == sample].copy()
        for strategy in TARGET_STRATEGIES:
            g = sub[sub["strategy"] == strategy].sort_values("return_date")
            if len(g) == 0:
                continue
            ax.plot(g["return_date"], g["wealth"], label=strategy)
        ax.set_title(f"FF25 Rank {rank} - Cumulative Wealth ({sample})")
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative wealth")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        suffix = sample.lower().replace("+", "plus")
        fig.savefig(outdir / f"ff25_rank{rank}_cumwealth_{suffix}.png", dpi=160)
        plt.close(fig)

def merge_three_summaries(file_map, rank: int, gamma: float = 5.0) -> pd.DataFrame:
    is_sum = load_summary(file_map, rank, oos=False).rename(columns=lambda c: f"IS_{c}" if c != "strategy" else c)
    oos_sum = load_summary(file_map, rank, oos=True).rename(columns=lambda c: f"OOS_{c}" if c != "strategy" else c)
    full_sum = build_combined_summary(file_map, rank, gamma=gamma).rename(columns=lambda c: f"IS_OOS_{c}" if c != "strategy" else c)
    merged = is_sum.merge(oos_sum, on="strategy", how="outer").merge(full_sum, on="strategy", how="outer").sort_values("strategy").reset_index(drop=True)
    merged.insert(0, "rank", rank)
    return merged

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str, required=True, help="Zip file, directory with CSVs, or directory containing a zip")
    parser.add_argument("--out-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--gamma", type=float, default=5.0, help="Risk aversion used in CER calculation")
    args = parser.parse_args()

    input_path = Path(args.input_path)
    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    root, cleanup_dir = prepare_input_root(input_path)
    try:
        file_map = build_file_map(root)
        ranks = sorted(file_map.keys())
        if not ranks:
            raise FileNotFoundError("No ff25_rank*_*.csv files were found in the input path.")

        combined_summary_frames = []
        combined_panel_frames = []

        for rank in ranks:
            panel = build_monthly_panel(file_map, rank)
            panel.insert(0, "rank", rank)
            panel.to_csv(outdir / f"ff25_rank{rank}_monthly_panel_with_wealth.csv", index=False)

            summary = merge_three_summaries(file_map, rank, gamma=args.gamma)
            summary.to_csv(outdir / f"ff25_rank{rank}_summary_IS_OOS_full.csv", index=False)

            save_plot(panel.drop(columns=["rank"]), rank, outdir)

            combined_summary_frames.append(summary)
            combined_panel_frames.append(panel)

        all_summary = pd.concat(combined_summary_frames, axis=0, ignore_index=True)
        all_summary = all_summary.sort_values(["rank", "strategy"]).reset_index(drop=True)
        all_summary.to_csv(outdir / "ff25_all_ranks_summary_IS_OOS_full.csv", index=False)

        all_panel = pd.concat(combined_panel_frames, axis=0, ignore_index=True)
        all_panel = all_panel.sort_values(["rank", "sample", "strategy", "return_date"]).reset_index(drop=True)
        all_panel.to_csv(outdir / "ff25_all_ranks_monthly_panel_with_wealth.csv", index=False)

        print(f"Done. Saved outputs to: {outdir}")
        print(f"Ranks found: {ranks}")
    finally:
        if cleanup_dir is not None and cleanup_dir.exists():
            shutil.rmtree(cleanup_dir, ignore_errors=True)

if __name__ == "__main__":
    main()
