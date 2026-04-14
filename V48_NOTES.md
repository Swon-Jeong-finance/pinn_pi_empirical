# dynalloc_v2 v48

## 핵심 변경
- validation 구간에서 rolling OOS estimation window 길이를 선택하는 stage 추가
- `rolling_selected_annual` 프로토콜 추가
- `rolling{N}m_annual`, `rolling{N}y_annual` 형태의 동적 rolling 프로토콜 지원
- rank sweep / replay가 manifest entry별 선택된 rolling window를 해석하도록 확장

## selection stage 변경
- `select-native-suite` 실행 시 기본적으로 validation protocol selection 활성화
- rolling grid 기본값: 60, 84, 120, 180, 240 months
- 선택 결과는 manifest entry / candidate metadata / selected yaml에 저장
- 새 CSV 출력:
  - `selection/selection_protocol_validation_summary.csv`
  - `selection/selection_protocol_validation_blocks.csv`

## 새 CLI 옵션
- `--rolling-oos-window-grid 60 84 120 180 240`
- `--disable-rolling-oos-window-selection`

## 새 manifest 기본 OOS 프로토콜
- `fixed`
- `expanding_annual`
- `rolling_selected_annual`

legacy manifest는 계속 `rolling20y_annual`도 지원.
