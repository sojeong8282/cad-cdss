"""CAD CDSS 최종 모델 분석 파이프라인.

이 파일은 기존 탐색용 분석 코드에서 최종적으로 채택한 분석 경로만
남긴 정리본이다.

분석 순서
---------
1. 원자료 로드 및 전처리 점검
2. 학습/테스트 고정 분리(242명/61명)
3. 학습 데이터에서 A, A+B, A+B+C 정보군 비교
4. A+B+C 53개 변수로 후보 모델 비교
5. 최종 후보의 확률 보정 상태 비교
6. GradientBoosting_Raw 반복 OOF 확률로 임계값 확인
7. 이미 확정한 임계값 0.64로 테스트셋을 단 한 번 평가
8. 최종 테스트 성능의 환자 단위 부트스트랩 95% 신뢰구간 계산
9. 앱에서 불러올 최종 모델 번들 저장

중요 원칙
---------
* 결측치 대체와 표준화는 Pipeline 안에서 학습 데이터로만 적합한다.
* 테스트셋은 모델, 변수군, 확률 보정 방법, 임계값 선택에 사용하지 않는다.
* 테스트 결과를 확인한 뒤 모델이나 임계값을 변경하지 않는다.
* 이 결과는 교육·연구용 CDSS 프로토타입이며 외부 검증 전에는 진단이나
  CAG 시행 여부 결정에 단독으로 사용할 수 없다.

필요 패키지
-----------
pip install numpy pandas scikit-learn imbalanced-learn openpyxl matplotlib joblib

실행 예시
---------
python cad_model_analysis_final.py
python cad_model_analysis_final.py --quick
python cad_model_analysis_final.py --stage final-only
python cad_model_analysis_final.py --data-path "C:/경로/Z-Alizadeh sani dataset.xlsx"
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "cad_cdss_matplotlib"),
)

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from imblearn.over_sampling import RandomOverSampler, SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline

    IMBLEARN_AVAILABLE = True
except ImportError:
    RandomOverSampler = None
    SMOTE = None
    ImbPipeline = None
    IMBLEARN_AVAILABLE = False

warnings.filterwarnings("ignore")


# =============================================================================
# 0. 확정 설정
# =============================================================================


@dataclass(frozen=True)
class AnalysisConfig:
    """한 번 확정한 분석 조건을 한곳에 고정한다."""

    data_path: Path
    output_dir: Path
    sheet_name: str = "Sheet 1 - Table 1"
    target_original_col: str = "Cath"
    target_col: str = "CAD"
    random_state: int = 42
    test_size: float = 0.20
    cv_splits: int = 5
    cv_repeats: int = 10
    bootstrap_iterations: int = 2_000
    paired_bootstrap_iterations: int = 5_000
    test_bootstrap_iterations: int = 5_000
    minimum_recall: float = 0.90
    fixed_threshold: float = 0.64
    expected_feature_count: int = 53
    final_model_name: str = "GradientBoosting_Raw"
    final_feature_group: str = "A+B+C"


VARIABLE_NAME_KR = {
    "Age": "나이",
    "Sex": "성별",
    "BMI": "체질량지수",
    "DM": "당뇨병",
    "HTN": "고혈압",
    "Current Smoker": "현재흡연",
    "EX-Smoker": "과거흡연",
    "FH": "가족력",
    "Obesity": "비만",
    "DLP": "이상지질혈증",
    "CRF": "만성신부전",
    "CVA": "뇌혈관질환",
    "CHF": "울혈성심부전",
    "Airway disease": "기도질환",
    "Thyroid Disease": "갑상선질환",
    "Typical Chest Pain": "전형적흉통",
    "Atypical": "비전형적흉통",
    "Nonanginal": "비협심증성통증",
    "Dyspnea": "호흡곤란",
    "LowTH Ang": "저역치협심증",
    "Function Class": "기능등급",
    "BP": "혈압",
    "PR": "맥박수",
    "Edema": "부종",
    "Weak Peripheral Pulse": "약한말초맥",
    "Lung rales": "폐수포음",
    "Systolic Murmur": "수축기잡음",
    "Diastolic Murmur": "이완기잡음",
    "Q Wave": "Q파",
    "St Elevation": "ST상승",
    "St Depression": "ST하강",
    "Tinversion": "T파역전",
    "LVH": "좌심실비대",
    "Poor R Progression": "R파진행불량",
    "BBB_LBBB": "좌각차단",
    "BBB_RBBB": "우각차단",
    "FBS": "공복혈당",
    "CR": "크레아티닌",
    "TG": "중성지방",
    "LDL": "저밀도지단백",
    "HDL": "고밀도지단백",
    "BUN": "혈중요소질소",
    "ESR": "적혈구침강속도",
    "HB": "헤모글로빈",
    "K": "칼륨",
    "Na": "나트륨",
    "WBC": "백혈구",
    "Lymph": "림프구",
    "Neut": "호중구",
    "PLT": "혈소판",
    "EF-TTE": "좌심실박출률",
    "Region RWMA": "국소벽운동이상",
    "VHD": "판막질환",
}


CDSS_GROUPS = {
    "Demographics": ["Age", "Sex", "BMI"],
    "RiskFactors": [
        "DM",
        "HTN",
        "Current Smoker",
        "EX-Smoker",
        "FH",
        "Obesity",
        "DLP",
        "CRF",
        "CVA",
        "CHF",
        "Airway disease",
        "Thyroid Disease",
    ],
    "Symptoms": [
        "Typical Chest Pain",
        "Atypical",
        "Nonanginal",
        "Dyspnea",
        "LowTH Ang",
        "Function Class",
    ],
    "PhysicalExam": [
        "BP",
        "PR",
        "Edema",
        "Weak Peripheral Pulse",
        "Lung rales",
        "Systolic Murmur",
        "Diastolic Murmur",
    ],
    "Laboratory": [
        "FBS",
        "CR",
        "TG",
        "LDL",
        "HDL",
        "BUN",
        "ESR",
        "HB",
        "K",
        "Na",
        "WBC",
        "Lymph",
        "Neut",
        "PLT",
    ],
    "ECG": [
        "Q Wave",
        "St Elevation",
        "St Depression",
        "Tinversion",
        "LVH",
        "Poor R Progression",
        "BBB_LBBB",
        "BBB_RBBB",
    ],
    "Imaging": ["EF-TTE", "Region RWMA", "VHD"],
}

ABC_COMPONENTS = {
    # A: 문진·과거력·증상·활력징후·신체검진
    "A": (
        CDSS_GROUPS["Demographics"]
        + CDSS_GROUPS["RiskFactors"]
        + CDSS_GROUPS["Symptoms"]
        + CDSS_GROUPS["PhysicalExam"]
    ),
    # B: 혈액검사 + ECG. 원자료에는 독립적인 X-ray 변수가 없다.
    "B": CDSS_GROUPS["Laboratory"] + CDSS_GROUPS["ECG"],
    # C: ECHO
    "C": CDSS_GROUPS["Imaging"],
}

ABC_COMBINATIONS = {
    "A": ABC_COMPONENTS["A"],
    "A+B": ABC_COMPONENTS["A"] + ABC_COMPONENTS["B"],
    "A+B+C": ABC_COMPONENTS["A"] + ABC_COMPONENTS["B"] + ABC_COMPONENTS["C"],
}


# =============================================================================
# 1. 데이터 로드와 전처리
# =============================================================================


def load_raw_data(config: AnalysisConfig) -> pd.DataFrame:
    if not config.data_path.exists():
        raise FileNotFoundError(
            f"데이터 파일을 찾을 수 없습니다: {config.data_path}\n"
            "--data-path 옵션으로 실제 위치를 지정하세요."
        )
    data = pd.read_excel(config.data_path, sheet_name=config.sheet_name)
    data.columns = data.columns.str.strip()
    return data


def add_explanation_categories(data: pd.DataFrame) -> pd.DataFrame:
    """화면 설명용 파생변수이며 모델 입력에는 사용하지 않는다."""

    result = data.copy()
    result["Age_cat"] = pd.cut(
        result["Age"], [0, 45, 60, 75, 120], labels=["<45", "45-59", "60-74", "75+"], right=False
    )
    result["BMI_cat"] = pd.cut(
        result["BMI"], [0, 18.5, 25, 30, 100], labels=["저체중", "정상/과체중", "비만1", "비만2+"], right=False
    )
    result["BP_cat"] = pd.cut(
        result["BP"], [0, 120, 140, 300], labels=["정상범위", "주의", "고혈압범위"], right=False
    )
    result["FBS_cat"] = pd.cut(
        result["FBS"], [0, 100, 126, 1000], labels=["정상", "공복혈당장애", "당뇨범위"], right=False
    )
    return result


def preprocess_for_cdss(
    raw_data: pd.DataFrame,
    config: AnalysisConfig,
) -> tuple[pd.DataFrame, list[str]]:
    """모델 전 결측치 대체 없이 코딩과 명백한 중복/상수만 정리한다."""

    data = raw_data.copy()
    dropped_columns: list[str] = []

    if config.target_original_col not in data.columns:
        raise ValueError(f"타깃 컬럼이 없습니다: {config.target_original_col}")

    data[config.target_col] = data[config.target_original_col].map({"Cad": 1, "Normal": 0})
    if data[config.target_col].isna().any():
        unknown = data.loc[data[config.target_col].isna(), config.target_original_col].unique()
        raise ValueError(f"알 수 없는 Cath 값: {unknown.tolist()}")
    data = data.drop(columns=[config.target_original_col])
    dropped_columns.append(config.target_original_col)

    yes_no_columns = [
        "Obesity",
        "CRF",
        "CVA",
        "Airway disease",
        "Thyroid Disease",
        "CHF",
        "DLP",
        "Weak Peripheral Pulse",
        "Lung rales",
        "Systolic Murmur",
        "Diastolic Murmur",
        "Dyspnea",
        "Atypical",
        "Nonanginal",
        "Exertional CP",
        "LowTH Ang",
        "LVH",
        "Poor R Progression",
    ]
    for column in yes_no_columns:
        if column in data.columns:
            data[column] = data[column].replace({"Y": 1, "N": 0})

    if "Sex" in data.columns:
        data["Sex"] = data["Sex"].replace({"Male": 1, "Female": 0, "Fmale": 0})

    if "VHD" in data.columns:
        data["VHD"] = data["VHD"].replace(
            {"N": 0, "mild": 1, "Moderate": 2, "Severe": 3}
        )

    if "BBB" in data.columns:
        data["BBB_LBBB"] = (data["BBB"] == "LBBB").astype(int)
        data["BBB_RBBB"] = (data["BBB"] == "RBBB").astype(int)
        data = data.drop(columns=["BBB"])
        dropped_columns.append("BBB")

    data = add_explanation_categories(data)

    # BMI와 Obesity는 서로 다른 원자료 변수이므로 둘 다 유지한다.
    # BMI와 직접 중복되는 Weight/Length만 제거한다.
    for column in ["Weight", "Length"]:
        if column in data.columns:
            data = data.drop(columns=[column])
            dropped_columns.append(column)

    explanation_columns = {"Age_cat", "BMI_cat", "BP_cat", "FBS_cat"}
    for column in data.columns:
        if column in explanation_columns:
            continue
        converted = pd.to_numeric(data[column], errors="coerce")
        if converted.isna().sum() == data[column].isna().sum():
            data[column] = converted

    candidate_columns = [
        column
        for column in data.columns
        if column != config.target_col and column not in explanation_columns
    ]
    constant_columns = [
        column for column in candidate_columns if data[column].nunique(dropna=False) <= 1
    ]
    if constant_columns:
        data = data.drop(columns=constant_columns)
        dropped_columns.extend(constant_columns)

    return data, dropped_columns


def get_feature_columns(data: pd.DataFrame, config: AnalysisConfig) -> list[str]:
    explanation_columns = {"Age_cat", "BMI_cat", "BP_cat", "FBS_cat"}
    return [
        column
        for column in data.columns
        if column != config.target_col
        and column not in explanation_columns
        and pd.api.types.is_numeric_dtype(data[column])
    ]


def validate_feature_schema(feature_columns: list[str], config: AnalysisConfig) -> None:
    expected = ABC_COMBINATIONS["A+B+C"]
    missing = [column for column in expected if column not in feature_columns]
    unexpected = [column for column in feature_columns if column not in expected]

    if missing or unexpected:
        raise ValueError(
            "최종 A+B+C 변수 정의와 전처리 결과가 일치하지 않습니다.\n"
            f"누락: {missing}\n예상 밖 변수: {unexpected}"
        )
    if len(feature_columns) != config.expected_feature_count:
        raise ValueError(
            f"최종 변수 수가 {config.expected_feature_count}개가 아닙니다: "
            f"{len(feature_columns)}개"
        )


def make_preprocessing_report(
    raw_data: pd.DataFrame,
    processed_data: pd.DataFrame,
    feature_columns: list[str],
    dropped_columns: list[str],
    config: AnalysisConfig,
) -> pd.DataFrame:
    rows = []
    for position, column in enumerate(feature_columns, start=1):
        rows.append(
            {
                "feature_order": position,
                "feature": column,
                "feature_kr": VARIABLE_NAME_KR.get(column, column),
                "dtype": str(processed_data[column].dtype),
                "missing_count": int(processed_data[column].isna().sum()),
                "missing_rate": float(processed_data[column].isna().mean()),
                "unique_count": int(processed_data[column].nunique(dropna=True)),
                "abc_group": next(
                    name for name, columns in ABC_COMPONENTS.items() if column in columns
                ),
            }
        )
    report = pd.DataFrame(rows)
    report.attrs["raw_shape"] = raw_data.shape
    report.attrs["processed_shape"] = processed_data.shape
    report.attrs["dropped_columns"] = dropped_columns
    report.to_csv(
        config.output_dir / "preprocessing_feature_report.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return report


def split_development_and_test(
    data: pd.DataFrame,
    feature_columns: list[str],
    config: AnalysisConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X = data.loc[:, feature_columns]
    y = data[config.target_col].astype(int)
    return train_test_split(
        X,
        y,
        test_size=config.test_size,
        stratify=y,
        random_state=config.random_state,
    )


# =============================================================================
# 2. 파이프라인과 후보 모델
# =============================================================================


def build_preprocessor(feature_columns: list[str]) -> ColumnTransformer:
    numeric_pipeline = SklearnPipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        transformers=[("numeric", numeric_pipeline, feature_columns)],
        remainder="drop",
    )


def make_pipeline(feature_columns: list[str], model, sampler=None):
    preprocessor = build_preprocessor(feature_columns)
    if sampler is None:
        return SklearnPipeline([("preprocess", preprocessor), ("model", model)])
    if not IMBLEARN_AVAILABLE:
        raise RuntimeError(
            "RandomOverSampler/SMOTE 분석에는 imbalanced-learn이 필요합니다.\n"
            "설치: python -m pip install imbalanced-learn"
        )
    return ImbPipeline(
        [("preprocess", preprocessor), ("sampler", sampler), ("model", model)]
    )


def get_model_candidates(feature_columns: list[str], random_state: int) -> dict[str, object]:
    candidates = {
        "LogisticRegression": LogisticRegression(
            max_iter=3_000, solver="lbfgs", random_state=random_state
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=500, min_samples_leaf=3, random_state=random_state, n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingClassifier(random_state=random_state),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=500, min_samples_leaf=3, random_state=random_state, n_jobs=-1
        ),
        "LogisticRegression_class_weight": LogisticRegression(
            max_iter=3_000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=random_state,
        ),
        "RandomForest_class_weight": RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
        "ExtraTrees_class_weight": ExtraTreesClassifier(
            n_estimators=500,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
        "SVM_RBF_class_weight": SVC(
            kernel="rbf", probability=True, class_weight="balanced", random_state=random_state
        ),
    }
    pipelines = {
        name: make_pipeline(feature_columns, model) for name, model in candidates.items()
    }

    if IMBLEARN_AVAILABLE:
        sampler_models = {
            "LogisticRegression": LogisticRegression(
                max_iter=3_000, solver="lbfgs", random_state=random_state
            ),
            "RandomForest": RandomForestClassifier(
                n_estimators=500,
                min_samples_leaf=3,
                random_state=random_state,
                n_jobs=-1,
            ),
            "GradientBoosting": GradientBoostingClassifier(random_state=random_state),
        }
        samplers = {
            "ROS": RandomOverSampler(random_state=random_state),
            "SMOTE": SMOTE(random_state=random_state, k_neighbors=5),
        }
        for sampler_name, sampler in samplers.items():
            for model_name, model in sampler_models.items():
                pipelines[f"{model_name}_{sampler_name}"] = make_pipeline(
                    feature_columns, model, sampler
                )
    return pipelines


def make_rf_ros_pipeline(feature_columns: list[str], random_state: int):
    if not IMBLEARN_AVAILABLE:
        raise RuntimeError("A·B·C 비교에는 imbalanced-learn이 필요합니다.")
    return make_pipeline(
        feature_columns,
        RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=3,
            random_state=random_state,
            n_jobs=-1,
        ),
        RandomOverSampler(random_state=random_state),
    )


def make_gradient_boosting_pipeline(feature_columns: list[str], random_state: int):
    return make_pipeline(
        feature_columns,
        GradientBoostingClassifier(random_state=random_state),
    )


# =============================================================================
# 3. 공통 평가 함수
# =============================================================================


def bootstrap_metric_ci(
    y_true: np.ndarray,
    y_probability: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    iterations: int,
    random_state: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(random_state)
    scores: list[float] = []
    sample_count = len(y_true)
    for _ in range(iterations):
        indices = rng.integers(0, sample_count, size=sample_count)
        sampled_y = y_true[indices]
        if np.unique(sampled_y).size < 2:
            continue
        try:
            scores.append(float(metric(sampled_y, y_probability[indices])))
        except ValueError:
            continue
    if not scores:
        return np.nan, np.nan
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def repeated_oof_evaluation(
    X: pd.DataFrame,
    y: pd.Series,
    specifications: dict[str, tuple[list[str], object]],
    config: AnalysisConfig,
    cv_repeats: int,
    bootstrap_iterations: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    """모든 비교 대상에 동일한 반복 층화 CV 분할을 적용한다."""

    repeated_cv = RepeatedStratifiedKFold(
        n_splits=config.cv_splits,
        n_repeats=cv_repeats,
        random_state=config.random_state,
    )
    splits = list(repeated_cv.split(X, y))
    y_array = y.to_numpy(dtype=int)
    fold_rows: list[dict] = []
    summary_rows: list[dict] = []
    oof_probabilities: dict[str, np.ndarray] = {}

    for number, (name, (feature_columns, base_pipeline)) in enumerate(
        specifications.items()
    ):
        print(f"\n[반복 OOF] {name} ({len(feature_columns)}개 변수)")
        probability_sum = np.zeros(len(y), dtype=float)
        prediction_count = np.zeros(len(y), dtype=int)
        current_fold_rows: list[dict] = []

        for split_number, (train_indices, valid_indices) in enumerate(splits):
            model = clone(base_pipeline)
            model.fit(
                X.iloc[train_indices].loc[:, feature_columns],
                y.iloc[train_indices],
            )
            probability = model.predict_proba(
                X.iloc[valid_indices].loc[:, feature_columns]
            )[:, 1]
            probability_sum[valid_indices] += probability
            prediction_count[valid_indices] += 1
            valid_y = y.iloc[valid_indices]
            current_fold_rows.append(
                {
                    "name": name,
                    "repeat": split_number // config.cv_splits + 1,
                    "fold": split_number % config.cv_splits + 1,
                    "n_features": len(feature_columns),
                    "roc_auc": roc_auc_score(valid_y, probability),
                    "pr_auc": average_precision_score(valid_y, probability),
                    "brier": brier_score_loss(valid_y, probability),
                }
            )

        if np.any(prediction_count == 0):
            raise RuntimeError(f"{name}: OOF 예측이 없는 환자가 있습니다.")

        mean_probability = probability_sum / prediction_count
        oof_probabilities[name] = mean_probability
        fold_rows.extend(current_fold_rows)
        current_folds = pd.DataFrame(current_fold_rows)
        roc_ci = bootstrap_metric_ci(
            y_array,
            mean_probability,
            roc_auc_score,
            bootstrap_iterations,
            config.random_state + number,
        )
        pr_ci = bootstrap_metric_ci(
            y_array,
            mean_probability,
            average_precision_score,
            bootstrap_iterations,
            config.random_state + 100 + number,
        )
        brier_ci = bootstrap_metric_ci(
            y_array,
            mean_probability,
            brier_score_loss,
            bootstrap_iterations,
            config.random_state + 200 + number,
        )
        summary_rows.append(
            {
                "name": name,
                "n_features": len(feature_columns),
                "fold_roc_auc_mean": current_folds["roc_auc"].mean(),
                "fold_roc_auc_sd": current_folds["roc_auc"].std(ddof=1),
                "fold_pr_auc_mean": current_folds["pr_auc"].mean(),
                "fold_pr_auc_sd": current_folds["pr_auc"].std(ddof=1),
                "fold_brier_mean": current_folds["brier"].mean(),
                "fold_brier_sd": current_folds["brier"].std(ddof=1),
                "oof_roc_auc": roc_auc_score(y_array, mean_probability),
                "roc_auc_ci_low": roc_ci[0],
                "roc_auc_ci_high": roc_ci[1],
                "oof_pr_auc": average_precision_score(y_array, mean_probability),
                "pr_auc_ci_low": pr_ci[0],
                "pr_auc_ci_high": pr_ci[1],
                "oof_brier": brier_score_loss(y_array, mean_probability),
                "brier_ci_low": brier_ci[0],
                "brier_ci_high": brier_ci[1],
            }
        )

    return pd.DataFrame(summary_rows), pd.DataFrame(fold_rows), oof_probabilities


def paired_bootstrap_difference(
    y_true: pd.Series,
    probability_a: np.ndarray,
    probability_b: np.ndarray,
    label: str,
    iterations: int,
    random_state: int,
) -> pd.DataFrame:
    """동일 환자를 함께 재표집하여 B-A 성능 차이를 계산한다."""

    y_array = y_true.to_numpy(dtype=int)
    metrics = {
        "roc_auc": roc_auc_score,
        "pr_auc": average_precision_score,
        "brier": brier_score_loss,
    }
    observed = {
        name: metric(y_array, probability_b) - metric(y_array, probability_a)
        for name, metric in metrics.items()
    }
    differences = {name: [] for name in metrics}
    rng = np.random.default_rng(random_state)

    for _ in range(iterations):
        indices = rng.integers(0, len(y_array), size=len(y_array))
        sampled_y = y_array[indices]
        if np.unique(sampled_y).size < 2:
            continue
        for name, metric in metrics.items():
            differences[name].append(
                metric(sampled_y, probability_b[indices])
                - metric(sampled_y, probability_a[indices])
            )

    rows = []
    for metric_name, values in differences.items():
        values_array = np.asarray(values, dtype=float)
        ci_low, ci_high = np.percentile(values_array, [2.5, 97.5])
        if metric_name == "brier":
            probability_improved = float(np.mean(values_array < 0))
            conclusion = (
                "통계적으로 명확한 개선"
                if ci_high < 0
                else "통계적으로 명확한 악화"
                if ci_low > 0
                else "차이 불확실"
            )
        else:
            probability_improved = float(np.mean(values_array > 0))
            conclusion = (
                "통계적으로 명확한 개선"
                if ci_low > 0
                else "통계적으로 명확한 악화"
                if ci_high < 0
                else "차이 불확실"
            )
        rows.append(
            {
                "comparison": label,
                "metric": metric_name,
                "difference_b_minus_a": observed[metric_name],
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
                "probability_improved": probability_improved,
                "conclusion": conclusion,
                "n_bootstrap": iterations,
            }
        )
    return pd.DataFrame(rows)


# =============================================================================
# 4. A·B·C와 후보 모델 비교
# =============================================================================


def compare_abc_groups(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    all_feature_columns: list[str],
    config: AnalysisConfig,
    cv_repeats: int,
    bootstrap_iterations: int,
    paired_bootstrap_iterations: int,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    specifications = {}
    for group_name, requested_columns in ABC_COMBINATIONS.items():
        requested_set = set(requested_columns)
        group_columns = [
            column for column in all_feature_columns if column in requested_set
        ]
        specifications[group_name] = (
            group_columns,
            make_rf_ros_pipeline(group_columns, config.random_state),
        )

    summary, folds, probabilities = repeated_oof_evaluation(
        X_train,
        y_train,
        specifications,
        config,
        cv_repeats,
        bootstrap_iterations,
    )
    summary["delta_roc_auc"] = summary["oof_roc_auc"].diff()
    summary["delta_pr_auc"] = summary["oof_pr_auc"].diff()
    summary["delta_brier"] = summary["oof_brier"].diff()
    summary = summary.rename(columns={"name": "group"})
    folds = folds.rename(columns={"name": "group"})

    paired = pd.concat(
        [
            paired_bootstrap_difference(
                y_train,
                probabilities["A"],
                probabilities["A+B"],
                "A → A+B (B 추가)",
                paired_bootstrap_iterations,
                config.random_state,
            ),
            paired_bootstrap_difference(
                y_train,
                probabilities["A+B"],
                probabilities["A+B+C"],
                "A+B → A+B+C (C/ECHO 추가)",
                paired_bootstrap_iterations,
                config.random_state + 1,
            ),
        ],
        ignore_index=True,
    )
    summary.to_csv(config.output_dir / "abc_repeated_cv_summary.csv", index=False, encoding="utf-8-sig")
    folds.to_csv(config.output_dir / "abc_repeated_cv_folds.csv", index=False, encoding="utf-8-sig")
    paired.to_csv(config.output_dir / "abc_paired_bootstrap_differences.csv", index=False, encoding="utf-8-sig")
    return summary, probabilities


def compare_candidate_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feature_columns: list[str],
    config: AnalysisConfig,
    cv_repeats: int,
    bootstrap_iterations: int,
    paired_bootstrap_iterations: int,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    candidates = get_model_candidates(feature_columns, config.random_state)
    specifications = {
        name: (feature_columns, pipeline) for name, pipeline in candidates.items()
    }
    summary, folds, probabilities = repeated_oof_evaluation(
        X_train,
        y_train,
        specifications,
        config,
        cv_repeats,
        bootstrap_iterations,
    )
    summary = summary.rename(columns={"name": "model"}).sort_values(
        ["oof_roc_auc", "oof_pr_auc", "oof_brier"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    summary.insert(0, "rank", np.arange(1, len(summary) + 1))
    folds = folds.rename(columns={"name": "model"})

    summary.to_csv(config.output_dir / "candidate_model_repeated_cv_summary.csv", index=False, encoding="utf-8-sig")
    folds.to_csv(config.output_dir / "candidate_model_repeated_cv_folds.csv", index=False, encoding="utf-8-sig")

    if "RandomForest_ROS" in probabilities and "GradientBoosting" in probabilities:
        paired = paired_bootstrap_difference(
            y_train,
            probabilities["GradientBoosting"],
            probabilities["RandomForest_ROS"],
            "GradientBoosting → RandomForest_ROS",
            paired_bootstrap_iterations,
            config.random_state,
        )
        paired.to_csv(
            config.output_dir / "rf_ros_vs_gradient_boosting_paired_bootstrap.csv",
            index=False,
            encoding="utf-8-sig",
        )
    return summary, probabilities


# =============================================================================
# 5. 확률 보정 평가
# =============================================================================


def calibration_intercept_slope(
    y_true: np.ndarray, y_probability: np.ndarray
) -> tuple[float, float]:
    clipped = np.clip(y_probability, 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=3_000)
    model.fit(logits, y_true)
    return float(model.intercept_[0]), float(model.coef_[0][0])


def calibration_bin_table(
    y_true: np.ndarray,
    y_probability: np.ndarray,
    name: str,
    n_bins: int = 10,
) -> pd.DataFrame:
    table = pd.DataFrame({"actual": y_true, "probability": y_probability})
    table["bin"] = pd.qcut(table["probability"], q=n_bins, duplicates="drop")
    result = (
        table.groupby("bin", observed=True)
        .agg(
            n_patients=("actual", "size"),
            mean_predicted_probability=("probability", "mean"),
            observed_cad_rate=("actual", "mean"),
            min_predicted_probability=("probability", "min"),
            max_predicted_probability=("probability", "max"),
        )
        .reset_index()
    )
    result.insert(0, "method", name)
    result["absolute_calibration_error"] = (
        result["observed_cad_rate"] - result["mean_predicted_probability"]
    ).abs()
    result["weighted_calibration_error"] = (
        result["n_patients"] / result["n_patients"].sum()
        * result["absolute_calibration_error"]
    )
    return result


def summarize_calibration_probabilities(
    y_true: pd.Series,
    probabilities: dict[str, np.ndarray],
    config: AnalysisConfig,
    file_prefix: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    y_array = y_true.to_numpy(dtype=int)
    rows = []
    bin_tables = []
    plot_data = {}

    for name, probability in probabilities.items():
        clipped = np.clip(probability, 1e-6, 1 - 1e-6)
        intercept, slope = calibration_intercept_slope(y_array, clipped)
        bins = calibration_bin_table(y_array, clipped, name)
        fraction_positive, mean_predicted = calibration_curve(
            y_array, clipped, n_bins=10, strategy="quantile"
        )
        rows.append(
            {
                "method": name,
                "roc_auc": roc_auc_score(y_array, clipped),
                "pr_auc": average_precision_score(y_array, clipped),
                "brier": brier_score_loss(y_array, clipped),
                "log_loss": log_loss(y_array, clipped, labels=[0, 1]),
                "ece": bins["weighted_calibration_error"].sum(),
                "maximum_calibration_error": bins["absolute_calibration_error"].max(),
                "calibration_intercept": intercept,
                "calibration_slope": slope,
            }
        )
        bin_tables.append(bins)
        plot_data[name] = (mean_predicted, fraction_positive)

    summary = pd.DataFrame(rows).sort_values(["brier", "log_loss", "ece"])
    all_bins = pd.concat(bin_tables, ignore_index=True)
    summary.to_csv(config.output_dir / f"{file_prefix}_summary.csv", index=False, encoding="utf-8-sig")
    all_bins.to_csv(config.output_dir / f"{file_prefix}_bins.csv", index=False, encoding="utf-8-sig")

    plt.figure(figsize=(8, 7))
    plt.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    for name, (mean_predicted, fraction_positive) in plot_data.items():
        plt.plot(mean_predicted, fraction_positive, marker="o", linewidth=2, label=name)
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed CAD proportion")
    plt.title("Calibration curve")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(config.output_dir / f"{file_prefix}_curve.png", dpi=200)
    plt.close()
    return summary, all_bins


def compare_gradient_boosting_calibration(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feature_columns: list[str],
    config: AnalysisConfig,
    cv_repeats: int,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """외부 반복 CV와 내부 3-fold 보정으로 Raw/Sigmoid/Isotonic을 비교한다."""

    X = X_train.loc[:, feature_columns].reset_index(drop=True)
    y = y_train.reset_index(drop=True).astype(int)
    methods = {
        "GradientBoosting_Raw": None,
        "GradientBoosting_Sigmoid": "sigmoid",
        "GradientBoosting_Isotonic": "isotonic",
    }
    probability_sums = {name: np.zeros(len(y), dtype=float) for name in methods}
    prediction_counts = {name: np.zeros(len(y), dtype=int) for name in methods}
    fold_rows = []
    outer_cv = RepeatedStratifiedKFold(
        n_splits=config.cv_splits,
        n_repeats=cv_repeats,
        random_state=config.random_state,
    )

    for split_number, (train_indices, valid_indices) in enumerate(outer_cv.split(X, y)):
        base = make_gradient_boosting_pipeline(feature_columns, config.random_state)
        X_fold_train = X.iloc[train_indices]
        X_fold_valid = X.iloc[valid_indices]
        y_fold_train = y.iloc[train_indices]
        y_fold_valid = y.iloc[valid_indices]

        raw = clone(base).fit(X_fold_train, y_fold_train)
        current = {"GradientBoosting_Raw": raw.predict_proba(X_fold_valid)[:, 1]}

        for name, method in methods.items():
            if method is None:
                continue
            inner_cv = StratifiedKFold(
                n_splits=3,
                shuffle=True,
                random_state=config.random_state + split_number + 1,
            )
            calibrated = CalibratedClassifierCV(
                estimator=clone(base), method=method, cv=inner_cv, ensemble=True
            )
            calibrated.fit(X_fold_train, y_fold_train)
            current[name] = calibrated.predict_proba(X_fold_valid)[:, 1]

        for name, probability in current.items():
            probability_sums[name][valid_indices] += probability
            prediction_counts[name][valid_indices] += 1
            fold_rows.append(
                {
                    "method": name,
                    "repeat": split_number // config.cv_splits + 1,
                    "fold": split_number % config.cv_splits + 1,
                    "roc_auc": roc_auc_score(y_fold_valid, probability),
                    "pr_auc": average_precision_score(y_fold_valid, probability),
                    "brier": brier_score_loss(y_fold_valid, probability),
                    "log_loss": log_loss(y_fold_valid, probability, labels=[0, 1]),
                }
            )

    averaged = {
        name: probability_sums[name] / prediction_counts[name] for name in methods
    }
    summary, _ = summarize_calibration_probabilities(
        y,
        averaged,
        config,
        "gradient_boosting_calibration",
    )
    pd.DataFrame(fold_rows).to_csv(
        config.output_dir / "gradient_boosting_calibration_folds.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return summary, averaged


# =============================================================================
# 6. OOF 임계값 확인
# =============================================================================


def build_threshold_table(
    y_true: pd.Series,
    y_probability: np.ndarray,
    minimum_recall: float,
) -> tuple[float, pd.DataFrame, dict]:
    y_array = y_true.to_numpy(dtype=int)
    rows = []
    for threshold in np.arange(0.01, 1.00, 0.01):
        prediction = (y_probability >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_array, prediction, labels=[0, 1]).ravel()
        rows.append(
            {
                "threshold": round(float(threshold), 2),
                "recall": tp / (tp + fn) if tp + fn else 0.0,
                "specificity": tn / (tn + fp) if tn + fp else 0.0,
                "precision": tp / (tp + fp) if tp + fp else 0.0,
                "negative_predictive_value": tn / (tn + fn) if tn + fn else 0.0,
                "f1": f1_score(y_array, prediction, zero_division=0),
                "accuracy": accuracy_score(y_array, prediction),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            }
        )
    table = pd.DataFrame(rows)
    eligible = table[table["recall"] >= minimum_recall].sort_values(
        ["specificity", "precision", "f1", "threshold"],
        ascending=[False, False, False, False],
    )
    if eligible.empty:
        raise RuntimeError(f"Recall {minimum_recall:.0%} 이상 임계값이 없습니다.")
    selected = eligible.iloc[0].to_dict()
    return float(selected["threshold"]), table, selected


def save_threshold_analysis(
    y_train: pd.Series,
    raw_oof_probability: np.ndarray,
    config: AnalysisConfig,
) -> float:
    reproduced_threshold, table, selected = build_threshold_table(
        y_train, raw_oof_probability, config.minimum_recall
    )
    table.to_csv(
        config.output_dir / "gradient_boosting_threshold_table.csv",
        index=False,
        encoding="utf-8-sig",
    )
    selected_record = {
        **selected,
        "reproduced_threshold": reproduced_threshold,
        "locked_final_threshold": config.fixed_threshold,
        "test_used_for_selection": False,
    }
    pd.DataFrame([selected_record]).to_csv(
        config.output_dir / "gradient_boosting_selected_threshold.csv",
        index=False,
        encoding="utf-8-sig",
    )

    plt.figure(figsize=(10, 6))
    plt.plot(table["threshold"], table["recall"], label="Recall")
    plt.plot(table["threshold"], table["specificity"], label="Specificity")
    plt.plot(table["threshold"], table["precision"], label="Precision", alpha=0.8)
    plt.axhline(config.minimum_recall, color="gray", linestyle="--", label="Minimum recall")
    plt.axvline(config.fixed_threshold, color="black", linestyle=":", label="Fixed threshold 0.64")
    plt.xlabel("Threshold")
    plt.ylabel("Metric")
    plt.xlim(0, 1)
    plt.ylim(0, 1.02)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(config.output_dir / "gradient_boosting_threshold_selection.png", dpi=200)
    plt.close()

    if not np.isclose(reproduced_threshold, config.fixed_threshold):
        print(
            "[주의] 현재 실행에서 재현된 OOF 임계값은 "
            f"{reproduced_threshold:.2f}이며, 확정 임계값은 "
            f"{config.fixed_threshold:.2f}입니다. 패키지 버전과 입력 파일을 확인하세요. "
            "최종 테스트에는 확정 임계값을 그대로 사용합니다."
        )
    return reproduced_threshold


# =============================================================================
# 7. 최종 테스트와 모델 저장
# =============================================================================


def classification_metrics(
    y_true: np.ndarray,
    y_probability: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    prediction = (y_probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    safe = lambda numerator, denominator: float(numerator / denominator) if denominator else np.nan
    return {
        "roc_auc": float(roc_auc_score(y_true, y_probability)),
        "pr_auc": float(average_precision_score(y_true, y_probability)),
        "brier": float(brier_score_loss(y_true, y_probability)),
        "log_loss": float(log_loss(y_true, y_probability, labels=[0, 1])),
        "accuracy": float(accuracy_score(y_true, prediction)),
        "recall": safe(tp, tp + fn),
        "specificity": safe(tn, tn + fp),
        "precision": safe(tp, tp + fp),
        "negative_predictive_value": safe(tn, tn + fn),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def evaluate_and_save_final_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_columns: list[str],
    config: AnalysisConfig,
) -> tuple[dict, np.ndarray]:
    """확정 모델·임계값으로 고정 테스트셋을 마지막에 한 번 평가한다."""

    model = make_gradient_boosting_pipeline(feature_columns, config.random_state)
    model.fit(X_train.loc[:, feature_columns], y_train)
    test_probability = model.predict_proba(X_test.loc[:, feature_columns])[:, 1]
    y_test_array = y_test.to_numpy(dtype=int)
    metrics = classification_metrics(y_test_array, test_probability, config.fixed_threshold)
    result = {
        "model": config.final_model_name,
        "calibration": "none",
        "variable_group": config.final_feature_group,
        "n_features": len(feature_columns),
        "threshold": config.fixed_threshold,
        "n_train": len(y_train),
        "n_test": len(y_test),
        "test_normal": int(np.sum(y_test_array == 0)),
        "test_cad": int(np.sum(y_test_array == 1)),
        **metrics,
    }
    pd.DataFrame([result]).to_csv(
        config.output_dir / "gradient_boosting_final_test_result.csv",
        index=False,
        encoding="utf-8-sig",
    )

    prediction = (test_probability >= config.fixed_threshold).astype(int)
    prediction_table = pd.DataFrame(
        {
            "source_index": X_test.index,
            "actual": y_test_array,
            "predicted_probability": test_probability,
            "fixed_threshold": config.fixed_threshold,
            "predicted_class": prediction,
            "correct": y_test_array == prediction,
        }
    )
    prediction_table.to_csv(
        config.output_dir / "gradient_boosting_final_test_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    serializable_config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in asdict(config).items()
    }
    bundle = {
        "model": model,
        "model_name": config.final_model_name,
        "calibration": "none",
        "threshold": config.fixed_threshold,
        "threshold_source": "training repeated OOF predictions",
        "threshold_selection_rule": "recall >= 0.90, then maximum specificity",
        "feature_group": config.final_feature_group,
        "feature_cols": list(feature_columns),
        "test_result": result,
        "config": serializable_config,
    }
    joblib.dump(bundle, config.output_dir / "cad_cdss_gradient_boosting_final.joblib")

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_name": config.final_model_name,
        "calibration": "none",
        "threshold": config.fixed_threshold,
        "threshold_source": "training repeated OOF predictions",
        "test_used_for_selection": False,
        "feature_group": config.final_feature_group,
        "n_features": len(feature_columns),
        "feature_columns": feature_columns,
        "test_result": result,
        "warning": (
            "교육·연구용 의료진 보조 CDSS 프로토타입. 외부기관 검증 전에는 "
            "진단 또는 CAG 시행 여부 결정에 단독으로 사용할 수 없음."
        ),
    }
    with (config.output_dir / "gradient_boosting_final_metadata.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    matrix = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
    plt.figure(figsize=(5, 4))
    plt.imshow(matrix, cmap="Blues")
    plt.title("Final Test Confusion Matrix\nGradientBoosting Raw")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks([0, 1], ["Normal", "CAD"])
    plt.yticks([0, 1], ["Normal", "CAD"])
    for row in range(2):
        for column in range(2):
            plt.text(column, row, int(matrix[row, column]), ha="center", va="center", fontsize=14)
    plt.tight_layout()
    plt.savefig(config.output_dir / "gradient_boosting_final_confusion_matrix.png", dpi=300)
    plt.close()
    return result, test_probability


def final_test_confidence_intervals(
    y_test: pd.Series,
    test_probability: np.ndarray,
    config: AnalysisConfig,
    iterations: int,
) -> pd.DataFrame:
    """고정 테스트 평가 뒤 불확실성만 계산하며 선택에는 사용하지 않는다."""

    y_array = y_test.to_numpy(dtype=int)
    point_estimates = classification_metrics(
        y_array, test_probability, config.fixed_threshold
    )
    metric_names = [
        "roc_auc",
        "pr_auc",
        "brier",
        "log_loss",
        "accuracy",
        "recall",
        "specificity",
        "precision",
        "negative_predictive_value",
        "f1",
    ]
    rng = np.random.default_rng(config.random_state)
    bootstrap_rows = []
    for bootstrap_number in range(iterations):
        indices = rng.integers(0, len(y_array), size=len(y_array))
        sampled_y = y_array[indices]
        if np.unique(sampled_y).size < 2:
            continue
        sampled = classification_metrics(
            sampled_y, test_probability[indices], config.fixed_threshold
        )
        bootstrap_rows.append(
            {"bootstrap_number": bootstrap_number + 1, **{name: sampled[name] for name in metric_names}}
        )
    bootstrap_table = pd.DataFrame(bootstrap_rows)
    directions = {
        "brier": "lower_is_better",
        "log_loss": "lower_is_better",
    }
    rows = []
    for name in metric_names:
        values = bootstrap_table[name].dropna().to_numpy(dtype=float)
        rows.append(
            {
                "metric": name,
                "estimate": point_estimates[name],
                "ci_low": float(np.percentile(values, 2.5)),
                "ci_high": float(np.percentile(values, 97.5)),
                "confidence_level": 0.95,
                "direction": directions.get(name, "higher_is_better"),
                "valid_bootstrap_samples": len(values),
                "method": "patient-level percentile bootstrap",
                "fixed_threshold": config.fixed_threshold,
                "used_for_model_or_threshold_selection": False,
            }
        )
    confidence_intervals = pd.DataFrame(rows)
    confidence_intervals.to_csv(
        config.output_dir / "gradient_boosting_final_test_confidence_intervals.csv",
        index=False,
        encoding="utf-8-sig",
    )
    bootstrap_table.to_csv(
        config.output_dir / "gradient_boosting_final_test_bootstrap_samples.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return confidence_intervals


# =============================================================================
# 8. 전체 실행
# =============================================================================


def class_distribution(y: pd.Series) -> str:
    normal = int((y == 0).sum())
    cad = int((y == 1).sum())
    return f"Normal={normal}, CAD={cad}, total={len(y)}"


def run_analysis(config: AnalysisConfig, stage: str, quick: bool) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    cv_repeats = 1 if quick else config.cv_repeats
    bootstrap_iterations = 100 if quick else config.bootstrap_iterations
    paired_iterations = 200 if quick else config.paired_bootstrap_iterations
    test_iterations = 200 if quick else config.test_bootstrap_iterations

    print("=" * 72)
    print("CAD CDSS 최종 모델 분석")
    print("=" * 72)
    raw_data = load_raw_data(config)
    data, dropped_columns = preprocess_for_cdss(raw_data, config)
    feature_columns = get_feature_columns(data, config)
    validate_feature_schema(feature_columns, config)
    preprocessing_report = make_preprocessing_report(
        raw_data, data, feature_columns, dropped_columns, config
    )
    data.to_csv(config.output_dir / "cad_cdss_preprocessed.csv", index=False, encoding="utf-8-sig")

    X_train, X_test, y_train, y_test = split_development_and_test(
        data, feature_columns, config
    )
    print(f"전체: {class_distribution(data[config.target_col])}")
    print(f"학습: {class_distribution(y_train)}")
    print(f"테스트(마지막까지 봉인): {class_distribution(y_test)}")
    print(f"최종 변수: {len(feature_columns)}개")
    print(f"제거 변수: {dropped_columns}")
    print(f"결측이 있는 최종 변수: {(preprocessing_report['missing_count'] > 0).sum()}개")

    reproduced_threshold = None
    if stage == "full":
        if not IMBLEARN_AVAILABLE:
            raise RuntimeError(
                "전체 비교에는 imbalanced-learn이 필요합니다.\n"
                "현재 환경에서 먼저 실행: python -m pip install imbalanced-learn\n"
                "최종 모델·테스트만 재현하려면 --stage final-only를 사용하세요."
            )

        print("\n[1/5] A·B·C 정보군 비교")
        compare_abc_groups(
            X_train,
            y_train,
            feature_columns,
            config,
            cv_repeats,
            bootstrap_iterations,
            paired_iterations,
        )

        print("\n[2/5] A+B+C 53개 변수 후보 모델 비교")
        candidate_summary, candidate_probabilities = compare_candidate_models(
            X_train,
            y_train,
            feature_columns,
            config,
            cv_repeats,
            bootstrap_iterations,
            paired_iterations,
        )
        candidate_calibration_probabilities = {
            name: candidate_probabilities[name]
            for name in ["RandomForest_ROS", "GradientBoosting"]
            if name in candidate_probabilities
        }
        summarize_calibration_probabilities(
            y_train,
            candidate_calibration_probabilities,
            config,
            "final_candidate_calibration",
        )
        print(candidate_summary.head(5).to_string(index=False))

        print("\n[3/5] GradientBoosting 보정 전·후 비교")
        _, calibration_probabilities = compare_gradient_boosting_calibration(
            X_train,
            y_train,
            feature_columns,
            config,
            cv_repeats,
        )

        print("\n[4/5] GradientBoosting_Raw OOF 임계값 확인")
        reproduced_threshold = save_threshold_analysis(
            y_train,
            calibration_probabilities["GradientBoosting_Raw"],
            config,
        )

    print("\n[5/5] 확정 모델·임계값으로 최종 테스트 및 95% CI")
    final_result, test_probability = evaluate_and_save_final_model(
        X_train,
        y_train,
        X_test,
        y_test,
        feature_columns,
        config,
    )
    confidence_intervals = final_test_confidence_intervals(
        y_test,
        test_probability,
        config,
        test_iterations,
    )

    run_summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "stage": stage,
        "quick_mode": quick,
        "data_path": str(config.data_path),
        "output_dir": str(config.output_dir),
        "n_total": len(data),
        "n_train": len(y_train),
        "n_test": len(y_test),
        "n_features": len(feature_columns),
        "feature_columns": feature_columns,
        "dropped_columns": dropped_columns,
        "final_model": config.final_model_name,
        "calibration": "none",
        "fixed_threshold": config.fixed_threshold,
        "reproduced_oof_threshold": reproduced_threshold,
        "test_used_for_selection": False,
        "final_test_result": final_result,
    }
    with (config.output_dir / "analysis_run_summary.json").open("w", encoding="utf-8") as file:
        json.dump(run_summary, file, ensure_ascii=False, indent=2)

    print("\n[완료]")
    print(f"결과 폴더: {config.output_dir}")
    print(pd.DataFrame([final_result]).to_string(index=False))
    print("\n95% 신뢰구간")
    print(
        confidence_intervals[["metric", "estimate", "ci_low", "ci_high"]].to_string(
            index=False, float_format=lambda value: f"{value:.3f}"
        )
    )


def parse_arguments() -> argparse.Namespace:
    script_directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="CAD CDSS 최종 모델 분석")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=script_directory / "Z-Alizadeh sani dataset.xlsx",
        help="원자료 Excel 경로",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_directory / "cad_cdss_final_outputs",
        help="분석 결과 저장 폴더",
    )
    parser.add_argument(
        "--stage",
        choices=["full", "final-only"],
        default="full",
        help="full: 전체 비교 재현, final-only: 확정 모델의 최종 평가만 재현",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="코드 동작 확인용 축소 반복(공식 결과로 사용 금지)",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    config = AnalysisConfig(
        data_path=arguments.data_path.resolve(),
        output_dir=arguments.output_dir.resolve(),
    )
    run_analysis(config, arguments.stage, arguments.quick)


if __name__ == "__main__":
    main()
