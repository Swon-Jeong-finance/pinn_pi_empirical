#!/usr/bin/env bash
# PI-PINN 하이퍼파라미터 sweep
# 사용법: bash tune_pipinn.sh <BASE_RESOLVED_CONFIG.yaml> [TUNE_ROOT]

set -euo pipefail

BASE_CFG="${1:-experiments/ff25_pls_factor_zoo_v2/rank_001/outputs/ff49_stage17_rank_sweep_cv2000_curve_core_pls_fixed_pls_H24_k2_rolling240m_annual_const_v2_apt_pipinn_rolling240m_annual/resolved_config.yaml}"
TUNE_ROOT="${2:-$(pwd)/pipinn_tune/$(date +%Y%m%d)}"
MAX_PARALLEL="${MAX_PARALLEL:-1}"   # ← 이 줄 추가 (환경변수로 제어 가능)
GPUS="${GPUS:-cuda:0,cuda:1}"     # ← 추가: 쉼표로 구분된 GPU 목록
mkdir -p "$TUNE_ROOT"

# GPU 리스트를 배열로
IFS=',' read -ra GPU_LIST <<< "$GPUS"
GPU_COUNT=${#GPU_LIST[@]}
GPU_CURSOR=0                       # round-robin 카운터

echo "[tune] base config : $BASE_CFG"
echo "[tune] output root : $TUNE_ROOT"

MANIFEST="$TUNE_ROOT/_manifest.tsv"

# PI-PINN이 받는 모든 하이퍼파라미터 20개를 positional로 받는다.
# 순서를 고정하기 위해 상수로 정의.
FIELDS=(outer_iters eval_epochs n_train_int n_train_bc n_val_int n_val_bc \
        p_uniform p_emp p_tau_head p_tau_near0 tau_head_window \
        lr grad_clip w_bc w_bc_dx \
        scheduler_factor scheduler_patience min_lr \
        width depth)



# manifest 헤더 (한 번만)
if [[ ! -f "$MANIFEST" ]]; then
  { printf "tag"; for f in "${FIELDS[@]}"; do printf "\t%s" "$f"; done; printf "\tdevice\toutput_dir\n"; } > "$MANIFEST"
fi

run_variant () {
  local tag=$1; shift
  if [[ $# -ne ${#FIELDS[@]} ]]; then
    echo "[error] $tag: expected ${#FIELDS[@]} args, got $#"
    return 1
  fi

  local vals=("$@")
  local out="$TUNE_ROOT/$tag"
  mkdir -p "$out"

  # 이미 완료됐으면 skip
  if [[ -f "$out/outputs/comparison_cross_modes_all_costs_summary.csv" ]] || \
     ls "$out/outputs"/*/comparison_cross_modes_all_costs_summary.csv > /dev/null 2>&1; then
    echo "[skip] $tag"
    return 0
  fi

  # 재시도 시 과거 실패 플래그 제거
  rm -f "$out/_FAILED"

  # Round-robin GPU 할당
  local device="${GPU_LIST[$((GPU_CURSOR % GPU_COUNT))]}"
  GPU_CURSOR=$((GPU_CURSOR + 1))

  # python으로 base YAML 복사 + PI-PINN 필드 override + snapshot 기록
  local cfg_path="$out/config.yaml"
  python3 - "$BASE_CFG" "$cfg_path" "$out" "$tag" "$device" "${vals[@]}" <<'PY'
import sys, json, pathlib, datetime, yaml

base_path, cfg_path, out_dir, tag, device, *vals = sys.argv[1:]  # ← device 받기
fields = ['outer_iters','eval_epochs','n_train_int','n_train_bc','n_val_int','n_val_bc',
          'p_uniform','p_emp','p_tau_head','p_tau_near0','tau_head_window',
          'lr','grad_clip','w_bc','w_bc_dx',
          'scheduler_factor','scheduler_patience','min_lr',
          'width','depth']
int_fields = {'outer_iters','eval_epochs','n_train_int','n_train_bc','n_val_int','n_val_bc',
              'tau_head_window','scheduler_patience','width','depth'}

cfg = yaml.safe_load(pathlib.Path(base_path).read_text())
cfg['project']['output_dir'] = f"{out_dir}/outputs"
cfg['project']['name']       = f"tune_{tag}"

p = cfg.setdefault('pipinn', {})
p['auto_output_subdir'] = False
p['device'] = device             # ← GPU 지정

overrides = {}
for k, v in zip(fields, vals):
    p[k] = int(v) if k in int_fields else float(v)
    overrides[k] = p[k]

pathlib.Path(cfg_path).write_text(yaml.safe_dump(cfg, sort_keys=False))

info = {
    'tag': tag,
    'timestamp': datetime.datetime.now().isoformat(timespec='seconds'),
    'base_config': base_path,
    'device': device,            # ← 메타데이터에도 기록
    'pipinn': overrides,
}
pathlib.Path(f"{out_dir}/run_info.json").write_text(json.dumps(info, indent=2))
PY

  # manifest 한 줄 (device 포함)
  { printf "%s" "$tag"; for v in "${vals[@]}"; do printf "\t%s" "$v"; done; printf "\t%s\t%s\n" "$device" "$out"; } >> "$MANIFEST"

  # 실행
  echo "[run ] $tag on $device"
  (
    if ! python3 -m dynalloc_v2.cli run --config "$cfg_path" \
          > "$out/stdout.log" 2> "$out/stderr.log"; then
      echo "[FAIL] $tag — check $out/stderr.log"
      touch "$out/_FAILED"
    else
      echo "[ok  ] $tag"
    fi
  ) &

  while (( $(jobs -rp | wc -l) >= MAX_PARALLEL )); do
    wait -n
  done
}

# =============================================================================
# 헬퍼: baseline 복제 + 주어진 key=value들만 덮어쓰는 방식
# 이렇게 하면 축마다 20개 인자를 다 쓸 필요 없이 변화점만 명시하면 됨
# =============================================================================
# 기준값 (선생님이 주신 YAML 그대로)
declare -A BASE=(
  [outer_iters]=10       [eval_epochs]=50
  [n_train_int]=4096     [n_train_bc]=1024
  [n_val_int]=2048       [n_val_bc]=512
  [p_uniform]=0.5        [p_emp]=0.5
  [p_tau_head]=0.5       [p_tau_near0]=0.2
  [tau_head_window]=0
  [lr]=0.0005            [grad_clip]=1.0
  [w_bc]=10.0            [w_bc_dx]=3.0
  [scheduler_factor]=0.5 [scheduler_patience]=3
  [min_lr]=1.0e-05
  [width]=128            [depth]=3
)

run () {
  # 사용: run <tag|auto> key1=val1 key2=val2 ...
  # tag를 'auto'로 주면 override 키=값들로 태그를 자동 생성
  local tag=$1; shift
  declare -A OVR=()
  for kv in "$@"; do
    OVR[${kv%%=*}]=${kv#*=}
  done
  # 태그 자동 생성
  if [[ "$tag" == "auto" ]]; then
    if [[ ${#OVR[@]} -eq 0 ]]; then
      tag="baseline"
    else
      # override 키를 정렬해서 재현성 확보 (같은 설정 → 같은 태그)
      local parts=()
      for k in $(printf '%s\n' "${!OVR[@]}" | sort); do
        local v=${OVR[$k]}
        # 소수점/마이너스/지수를 짧게 정리: 0.0005 → 5e-4, 1.0e-05 → 1e-5
        v=$(python3 -c "v='$v'; f=float(v); print(f'{f:g}'.replace('+0','+').replace('-0','-'))" 2>/dev/null || echo "$v")
        parts+=("${k}${v}")
      done
      tag=$(IFS=_; echo "${parts[*]}")
    fi
  fi
  # FIELDS 순서대로 인자 조립
  local args=()
  for f in "${FIELDS[@]}"; do
    if [[ -n "${OVR[$f]+x}" ]]; then
      args+=("${OVR[$f]}")
    else
      args+=("${BASE[$f]}")
    fi
  done
  run_variant "$tag" "${args[@]}"
}

# =============================================================================
# 여기부터 실제 sweep 정의
# 형식: run <tag> <field1>=<value1> <field2>=<value2> ...
# 명시하지 않은 필드는 위 BASE 값이 자동으로 쓰임
# =============================================================================

# (0) baseline 먼저 한 번
run  baseline

# (A) outer × epochs — 시간 예산
run  auto    outer_iters=10   eval_epochs=100
run  auto   outer_iters=10   eval_epochs=200
run  auto  outer_iters=15   eval_epochs=50
run  auto  outer_iters=20   eval_epochs=50
# run  auto  outer_iters=3   eval_epochs=100

# (B) 네트워크 크기 (width × depth)
run  auto    width=64   depth=2
run  auto   width=128  depth=1
run  auto   width=128  depth=2
# run  net_w192_d4   width=192  depth=4
run  auto   width=256  depth=2

# (C) 학습률 + 스케줄러
run  auto       lr=1.0e-3
run  auto       lr=7.0e-4
# run  auto       lr=5.0e-2
# run  sched_f03_p5  scheduler_factor=0.3  scheduler_patience=5
# run  sched_f07_p2  scheduler_factor=0.7  scheduler_patience=2

# (D) BC 가중치
# run  bc_10         w_bc=10.0  w_bc_dx=1.5
run  auto   w_bc=20.0  w_bc_dx=3.0
run  auto   w_bc=10.0  w_bc_dx=5.0
run  auto   w_bc=10.0  w_bc_dx=1.0   # terminal-dominant

# (E) 샘플링 mixture (uniform vs empirical)
run  auto  p_uniform=0.3  p_emp=0.7
# run  auto  p_uniform=0.5  p_emp=0.5
run  auto  p_uniform=0.7  p_emp=0.3

# (F) tau head/near0 샘플링
run  auto   p_tau_head=0.3  p_tau_near0=0.1
run  auto   p_tau_head=0.5  p_tau_near0=0.2
run  auto  p_tau_head=0.7  p_tau_near0=0.2
# run  tauhead_win6  p_tau_head=0.5  p_tau_near0=0.2  tau_head_window=6

# (G) 훈련/검증 콜로케이션 포인트 수
run  auto    n_train_int=2048  n_train_bc=512   n_val_int=1024  n_val_bc=256
run  auto   n_train_int=4096  n_train_bc=1024  n_val_int=4096  n_val_bc=1024
run  auto    n_train_int=8192  n_train_bc=2048  n_val_int=4096  n_val_bc=1024

# (H) gradient clipping
# run  auto       grad_clip=0.5
# run  auto       grad_clip=1.0
# run  auto       grad_clip=2.0

wait   # ← 이 줄 추가: 백그라운드 job이 전부 끝날 때까지 대기
echo ""
echo "[done] manifest: $MANIFEST"
echo "[done] 집계: python3 collect_results.py $TUNE_ROOT"

python3 "$(dirname "$0")/collect_results.py" "$TUNE_ROOT"