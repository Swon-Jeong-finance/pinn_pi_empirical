# dynalloc_v2 v49

## 핵심 변경
- protocol selection을 별도 post-hoc validation stage로 두지 않고, **stage1 후보축에 직접 통합**
- stage1 선택 단위가 이제 **spec-protocol pair**
  - 예: `pls_H12_k3 + fixed`, `pls_H12_k3 + expanding_annual`, `pls_H12_k3 + rolling60m_annual`
- stage2는 **stage1에서 살아남은 spec-protocol pair에만 covariance model**을 붙여 rerank
  - 예: `(pls_H12_k3 + rolling60m_annual) × {const,dcc,adcc,regime_dcc}`

## selection pipeline
- **stage1:** mean-first cheap screening over `spec × protocol`
- **stage2:** real PPGDPO-lite rerank over `(selected spec × protocol) × covariance`
- 별도의 `validation_protocol_selection` 단계는 비활성화되고, manifest에는 참고용 메타데이터만 남김

## 새 기본 동작
- `select-native-suite` 기본 protocol candidates:
  - `fixed`
  - `expanding_annual`
  - `rolling{N}m_annual` for values in `--rolling-oos-window-grid` when rolling selection is enabled
- 결과 manifest의 기본 OOS protocol은 `selected_protocol`
  - 각 entry가 자기에게 선택된 실제 protocol (`fixed` / `expanding_annual` / `rollingNm_annual`)을 들고 있음
- rank sweep / replay는 `selected_protocol`을 해석해서 entry별 실제 protocol로 실행

## 새 CLI 옵션
- `--selection-protocols fixed expanding_annual rolling60m_annual rolling84m_annual`
  - stage1/stage2에 직접 넣을 protocol 후보를 명시적으로 지정

## 구현상 참고
- 기존 `rolling_selected_annual`은 backward compatibility 차원에서 계속 해석 가능
- `tests/test_native_selection.py`의 일부 오래된 assertion은 v48의 post-hoc rolling validation semantics를 기대하므로 그대로는 안 맞을 수 있음
