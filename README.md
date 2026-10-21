# Predicting Airline Satisfaction (Kaggle Playground S6E10)

승객 정보와 기내 서비스 평점으로 항공사 만족 여부를 예측하는 이진 분류 문제입니다. 평가 지표는 ROC-AUC입니다.

- 대회: https://www.kaggle.com/competitions/playground-series-s6e10

## 결과

| 제출 | 모델 | CV (OOF AUC) | Public LB |
|---|---|---|---|
| 1 | LightGBM 5-Fold | 0.95866 | 0.95802 |
| 2 | LightGBM 5-Fold + 규제 | 0.95909 | **0.95853** |

## 실행 방법

1. 대회 페이지에서 데이터를 내려받아 `data/`에 `train.csv`, `test.csv`, `sample_submission.csv`를 넣습니다. 데이터는 저장소에 포함되어 있지 않습니다.
2. 환경을 만들고 노트북을 엽니다.

```bash
uv sync
uv run jupyter lab
```

## 노트북

| 파일 | 내용 |
|---|---|
| `notebooks/baseline.ipynb` | EDA, 로지스틱 회귀, LightGBM 5-Fold |
| `notebooks/exp01_feature_ablation.ipynb` | 변수 제거 |
| `notebooks/exp02_rating_features.ipynb` | 평점 요약 변수 추가 |
| `notebooks/exp03_interaction_features.ipynb` | 조합 변수 추가 |
| `notebooks/exp04_lgbm_params.ipynb` | LightGBM 파라미터 (규제) |
| `notebooks/exp05_ensemble.ipynb` | LightGBM + XGBoost |
| `notebooks/exp06_mlp.ipynb` | MLP (Kaggle GPU에서 실행) |
| `notebooks/exp07_blend.ipynb` | LightGBM + XGBoost + MLP 블렌딩 |
| `notebooks/submission_v2.ipynb` | 제출 2 파일 생성 |

`exp07`은 `exp06`이 만든 `outputs/oof_mlp.csv`, `outputs/test_mlp.csv`가 필요합니다. 이 파일들도 저장소에 포함되어 있지 않으므로, `exp06`을 Kaggle에서 실행해 생성해야 합니다.
