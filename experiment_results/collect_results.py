#!/usr/bin/env python3
"""
Sweep 디렉토리 전체를 스캔해서 estimated vs zero를 양옆에 붙인 비교 테이블을 만든다.
정렬하지 않고 run 순서(run_info.json의 timestamp)대로 배치한다.

사용법: python3 collect_results.py <TUNE_ROOT>
"""

import json
import sys
from pathlib import Path

import pandas as pd


def find_summary(run_dir: Path) -> Path | None:
    hits = list(run_dir.rglob('comparison_cross_modes_all_costs_summary.csv'))
    return hits[0] if hits else None

def find_training_logs(run_dir: Path) -> list[Path]:
    """
    PI-PINN training log CSV들을 모두 찾는다.
    파일명 패턴: refit_XXX_YYYYMMDD.csv
    run_dir 안의 outputs/ 아래 training_logs/pipinn/ 에 위치.
    """
    return sorted(run_dir.rglob('refit_*.csv'))


def summarize_training_diagnostics(run_dir: Path) -> dict[str, float]:
    """
    해당 run의 모든 refit log를 읽어 진단 지표의 요약 통계를 낸다.

    각 refit에서 마지막 epoch 값을 뽑아 refit들 간 평균/중앙값을 계산.
    PDE loss도 함께 기록 (수렴도 체크용).

    반환 필드:
      - final_val_pde_mean:  refit별 마지막 val_pde의 평균
      - final_grad_l2_mean:  refit별 마지막 val_grad_l2_mean의 평균
      - final_grad_l2_cv:    refit별 마지막 val_grad_l2_cv의 평균
      - final_grad_dir_cos:  refit별 마지막 val_grad_dir_cos_mean의 평균 (핵심 지표)
      - n_refits:            집계된 refit 수
    """
    log_paths = find_training_logs(run_dir)
    empty = {'final_val_pde_mean': float('nan'),
             'final_grad_l2_mean': float('nan'),
             'final_grad_l2_cv': float('nan'),
             'final_grad_dir_cos': float('nan'),
             'n_refits': 0}
    if not log_paths:
        return empty

    # 필수 컬럼이 있는지도 체크 (예전 log는 진단 컬럼 없을 수 있음)
    diag_cols = ['val_pde', 'val_grad_l2_mean', 'val_grad_l2_cv', 'val_grad_dir_cos_mean']
    per_refit = []
    for p in log_paths:
        try:
            df = pd.read_csv(p)
            if len(df) == 0:
                continue
            last = df.iloc[-1]
            row = {c: (float(last[c]) if c in df.columns and pd.notna(last[c]) else float('nan'))
                   for c in diag_cols}
            per_refit.append(row)
        except Exception:
            continue

    if not per_refit:
        return empty

    agg = pd.DataFrame(per_refit)

    def _mean(col: str) -> float:
        if col not in agg.columns:
            return float('nan')
        vals = pd.to_numeric(agg[col], errors='coerce')
        return float(vals.mean()) if vals.notna().any() else float('nan')

    return {
        'final_val_pde_mean':   _mean('val_pde'),
        'final_grad_l2_mean':   _mean('val_grad_l2_mean'),
        'final_grad_l2_cv':     _mean('val_grad_l2_cv'),
        'final_grad_dir_cos':   _mean('val_grad_dir_cos_mean'),
        'n_refits':             int(len(agg)),
    }

def pick_scalar(summary: pd.DataFrame, *, strategy_startswith: str,
                cross_mode: str, column: str) -> float:
    """PI-PINN/PPGDPO 백엔드 모두 대응 (strategy 접두사로 매칭)."""
    if column not in summary.columns:
        return float('nan')
    mask = summary['strategy'].astype(str).str.startswith(strategy_startswith)
    sub = summary.loc[mask & (summary['cross_mode'] == cross_mode), column]
    return float(sub.iloc[0]) if len(sub) and pd.notna(sub.iloc[0]) else float('nan')


def collect(tune_root: Path, *, backend_prefix: str = 'pipinn') -> pd.DataFrame:
    rows = []
    # run_info.json의 tag 순서(= 스크립트에서 실행된 순서)를 유지
    # mtime으로 정렬하면 실행 시각 순, 파일명으로 정렬하면 알파벳 순.
    # 여기서는 run_info.json 안의 timestamp 기준으로 정렬해서 실행 순서를 재현.
    run_infos = []
    for run_info_path in tune_root.glob('*/run_info.json'):
        info = json.loads(run_info_path.read_text())
        run_infos.append((info.get('timestamp', ''), run_info_path, info))
    run_infos.sort(key=lambda t: t[0])  # 실행 시간 순

    for _, run_info_path, info in run_infos:
        run_dir = run_info_path.parent
        tag = info['tag']
        hp = info.get('pipinn', {})

        row = {'tag': tag, **hp}

        if (run_dir / '_FAILED').exists():
            row['status'] = 'FAILED'
            rows.append(row)
            continue

        summary_path = find_summary(run_dir)
        if summary_path is None:
            row['status'] = 'NO_SUMMARY'
            rows.append(row)
            continue

        summary = pd.read_csv(summary_path)

        # 핵심: 각 metric에 대해 estimated/zero를 쌍으로 수집
        for metric in ['cer_ann', 'sharpe', 'max_drawdown', 'avg_turnover']:
            row[f'{metric}_est']  = pick_scalar(summary, strategy_startswith=backend_prefix,
                                                cross_mode='estimated', column=metric)
            row[f'{metric}_zero'] = pick_scalar(summary, strategy_startswith=backend_prefix,
                                                cross_mode='zero',      column=metric)
            # 격차 (estimated - zero); cer은 양수면 cross가 도움, MDD는 음수 친화적
            est  = row[f'{metric}_est']
            zer  = row[f'{metric}_zero']
            row[f'{metric}_diff'] = (est - zer) if pd.notna(est) and pd.notna(zer) else float('nan')

        # --- PI-PINN training 진단 지표 (spectral bias 감지) ---
        diagnostics = summarize_training_diagnostics(run_dir)
        row.update(diagnostics)

        row['status'] = 'OK'
        row['summary_path'] = str(summary_path.relative_to(tune_root))
        rows.append(row)

    df = pd.DataFrame(rows)

    # 컬럼 순서: [tag, status] + 하이퍼파라미터 + 쌍지표(est/zero/diff) + 경로
    hp_cols = list(df.columns.difference(
        ['tag', 'status', 'summary_path',
         'cer_ann_est','cer_ann_zero','cer_ann_diff',
         'sharpe_est','sharpe_zero','sharpe_diff',
         'max_drawdown_est','max_drawdown_zero','max_drawdown_diff',
         'avg_turnover_est','avg_turnover_zero','avg_turnover_diff']
    ))
    # 하이퍼파라미터 컬럼은 스크립트에 등장한 순서로 고정
    ordered_hp = [c for c in
        ['outer_iters','eval_epochs','n_train_int','n_train_bc','n_val_int','n_val_bc',
         'p_uniform','p_emp','p_tau_head','p_tau_near0','tau_head_window',
         'lr','grad_clip','w_bc','w_bc_dx',
         'scheduler_factor','scheduler_patience','min_lr','width','depth']
        if c in hp_cols]

    metric_cols = []
    for m in ['cer_ann', 'sharpe', 'max_drawdown', 'avg_turnover']:
        metric_cols += [f'{m}_est', f'{m}_zero', f'{m}_diff']
    metric_cols = [c for c in metric_cols if c in df.columns]

    # 진단 지표 컬럼 (있는 것만)
    diag_cols = [c for c in
                 ['final_val_pde_mean', 'final_grad_l2_mean', 'final_grad_l2_cv',
                  'final_grad_dir_cos', 'n_refits']
                 if c in df.columns]

    ordered = ['tag', 'status'] + ordered_hp + metric_cols + ['summary_path']
    ordered = [c for c in ordered if c in df.columns]
    df = df[ordered]
    return df


if __name__ == '__main__':
    tune_root = Path(sys.argv[1]).expanduser().resolve()
    backend = sys.argv[2] if len(sys.argv) > 2 else 'pipinn'

    df = collect(tune_root, backend_prefix=backend)
    out_csv = tune_root / 'sweep_comparison.csv'
    df.to_csv(out_csv, index=False)
    print(f'[saved] {out_csv}  ({len(df)} runs)')
    print()

    # 터미널 디스플레이: 하이퍼파라미터 요약 + 핵심 지표 + 진단 지표
    display = df[[c for c in
        ['tag','status',
         'cer_ann_est','cer_ann_zero','cer_ann_diff',
         'sharpe_est','sharpe_zero','sharpe_diff',
         # --- 진단 ---
         'final_grad_dir_cos','final_grad_l2_cv','final_val_pde_mean']
        if c in df.columns]]
    with pd.option_context(
        'display.max_rows', None,
        'display.width', 220,
        'display.float_format', lambda x: f'{x:+.4f}' if pd.notna(x) else '   nan'
    ):
        print(display.to_string(index=False))