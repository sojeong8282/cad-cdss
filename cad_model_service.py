"""CAD CDSS 앱용 예측 서비스.

이 모듈은 Streamlit 화면과 모델 분석 코드를 분리하기 위한 중간 계층이다.

담당하는 일
-----------
1. 최종 모델 번들(joblib) 1회 로드
2. 저장된 feature_cols 순서대로 53개 입력 정렬
3. 미입력값을 NaN으로 전달
4. 미입력 항목과 입력 완성도 보고
5. predict_proba() 실행 및 저장된 임계값 적용

담당하지 않는 일
----------------
* 모델 학습
* 교차검증
* 연구용 변수 조합 성능 비교
* 새로운 임계값 선택
* 화면 구성

중요
----
* 중앙값 대체와 표준화는 저장된 sklearn Pipeline 안에 포함되어 있다.
  이 파일에서 중앙값을 다시 계산하거나 별도로 표준화하면 안 된다.
* 반환되는 수치는 학습 데이터 패턴에 대한 모델 점수이다. 일반 인구에서의
  절대적인 CAD 발병확률이나 확진 결과로 해석하면 안 된다.
* joblib 파일은 임의의 코드를 포함할 수 있으므로 직접 생성하고 신뢰할 수 있는
  모델 파일만 로드해야 한다.

app.py 사용 예시
---------------
from cad_model_service import predict_cad

patient_data = {
    "Age": 68,
    "Sex": 1,
    "BMI": 25.1,
    # 나머지 모델 변수...
}

result = predict_cad(patient_data)
print(result["cad_score"])
print(result["risk_label"])
print(result["missing_features_kr"])
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd


# =============================================================================
# 0. 경로와 표시명
# =============================================================================


MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = (
    MODULE_DIR
    / "cad_cdss_final_outputs"
    / "cad_cdss_gradient_boosting_final.joblib"
)


FEATURE_NAME_KR = {
    "Age": "나이",
    "Sex": "성별",
    "BMI": "체질량지수",
    "DM": "당뇨병",
    "HTN": "고혈압",
    "Current Smoker": "현재 흡연",
    "EX-Smoker": "과거 흡연",
    "FH": "관상동맥질환 가족력",
    "Obesity": "비만",
    "CRF": "만성신부전",
    "CVA": "뇌혈관질환",
    "Airway disease": "기도질환",
    "Thyroid Disease": "갑상선질환",
    "CHF": "울혈성심부전",
    "DLP": "이상지질혈증",
    "BP": "수축기 혈압",
    "PR": "맥박수",
    "Edema": "부종",
    "Weak Peripheral Pulse": "약한 말초맥박",
    "Lung rales": "폐수포음",
    "Systolic Murmur": "수축기 잡음",
    "Diastolic Murmur": "이완기 잡음",
    "Typical Chest Pain": "전형적 흉통",
    "Dyspnea": "호흡곤란",
    "Function Class": "기능등급",
    "Atypical": "비전형적 흉통",
    "Nonanginal": "비협심증성 통증",
    "LowTH Ang": "저역치 협심증",
    "Q Wave": "Q파",
    "St Elevation": "ST 상승",
    "St Depression": "ST 하강",
    "Tinversion": "T파 역전",
    "LVH": "좌심실비대",
    "Poor R Progression": "R파 진행 불량",
    "FBS": "공복혈당",
    "CR": "크레아티닌",
    "TG": "중성지방",
    "LDL": "LDL 콜레스테롤",
    "HDL": "HDL 콜레스테롤",
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
    "BBB_LBBB": "좌각차단",
    "BBB_RBBB": "우각차단",
}


CLINICAL_FEATURE_GROUPS = {
    "문진·과거력·활력징후·신체검사": {
        "Age",
        "Sex",
        "BMI",
        "DM",
        "HTN",
        "Current Smoker",
        "EX-Smoker",
        "FH",
        "Obesity",
        "CRF",
        "CVA",
        "Airway disease",
        "Thyroid Disease",
        "CHF",
        "DLP",
        "Typical Chest Pain",
        "Atypical",
        "Nonanginal",
        "Dyspnea",
        "LowTH Ang",
        "Function Class",
        "BP",
        "PR",
        "Edema",
        "Weak Peripheral Pulse",
        "Lung rales",
        "Systolic Murmur",
        "Diastolic Murmur",
    },
    "혈액검사": {
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
    },
    "심전도": {
        "Q Wave",
        "St Elevation",
        "St Depression",
        "Tinversion",
        "LVH",
        "Poor R Progression",
        "BBB_LBBB",
        "BBB_RBBB",
    },
    "심초음파(ECHO)": {"EF-TTE", "Region RWMA", "VHD"},
}


BINARY_FEATURES = {
    "Sex",
    "DM",
    "HTN",
    "Current Smoker",
    "EX-Smoker",
    "FH",
    "Obesity",
    "CRF",
    "CVA",
    "Airway disease",
    "Thyroid Disease",
    "CHF",
    "DLP",
    "Edema",
    "Weak Peripheral Pulse",
    "Lung rales",
    "Systolic Murmur",
    "Diastolic Murmur",
    "Typical Chest Pain",
    "Dyspnea",
    "Atypical",
    "Nonanginal",
    "LowTH Ang",
    "Q Wave",
    "St Elevation",
    "St Depression",
    "Tinversion",
    "LVH",
    "Poor R Progression",
    "BBB_LBBB",
    "BBB_RBBB",
}


MISSING_TEXT_VALUES = {
    "",
    "-",
    "na",
    "n/a",
    "nan",
    "none",
    "null",
    "unknown",
    "미입력",
    "미선택",
    "미확인",
    "확인되지 않음",
    "결과 대기",
    "검사 안 함",
    "검사하지 않음",
}


TRUE_TEXT_VALUES = {
    "1",
    "y",
    "yes",
    "true",
    "있음",
    "예",
    "양성",
    "male",
    "남",
    "남성",
}


FALSE_TEXT_VALUES = {
    "0",
    "n",
    "no",
    "false",
    "없음",
    "아니오",
    "음성",
    "female",
    "fmale",
    "여",
    "여성",
}


# =============================================================================
# 1. 반환 자료구조
# =============================================================================


@dataclass(frozen=True)
class MissingValueReport:
    total_features: int
    entered_features: int
    missing_count: int
    completeness: float
    missing_features: list[str]
    missing_features_kr: list[str]
    missing_by_group: dict[str, list[str]]
    missing_by_group_kr: dict[str, list[str]]
    unknown_input_fields: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CadPredictionResult:
    cad_score: float
    threshold: float
    predicted_class: int
    risk_label: str
    model_name: str
    feature_group: str
    n_features: int
    missing_report: MissingValueReport
    warnings: list[str]
    interpretation: str

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        # app.py에서 자주 사용할 항목은 최상위에도 제공한다.
        result.update(self.missing_report.to_dict())
        result.pop("missing_report", None)
        return result


# =============================================================================
# 2. 서비스 클래스
# =============================================================================


class CadModelService:
    """저장된 최종 CAD 모델의 입력 검증과 예측을 담당한다."""

    REQUIRED_BUNDLE_KEYS = {
        "model",
        "model_name",
        "threshold",
        "feature_group",
        "feature_cols",
    }

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        *,
        low_completeness_threshold: float = 0.80,
    ) -> None:
        self.model_path = Path(model_path).expanduser().resolve()
        self.low_completeness_threshold = float(low_completeness_threshold)

        if not 0.0 <= self.low_completeness_threshold <= 1.0:
            raise ValueError("low_completeness_threshold는 0과 1 사이여야 합니다.")

        self.bundle = self._load_bundle(self.model_path)
        self.model = self.bundle["model"]
        self.model_name = str(self.bundle["model_name"])
        self.threshold = float(self.bundle["threshold"])
        self.feature_group = str(self.bundle["feature_group"])
        self.feature_cols = list(self.bundle["feature_cols"])

        self._validate_bundle()

    @classmethod
    def _load_bundle(cls, model_path: Path) -> dict[str, Any]:
        if not model_path.exists():
            raise FileNotFoundError(
                "최종 모델 파일을 찾을 수 없습니다.\n"
                f"확인 경로: {model_path}\n"
                "cad_cdss_final_outputs 폴더와 joblib 파일의 위치를 확인하세요."
            )

        loaded = joblib.load(model_path)
        if not isinstance(loaded, dict):
            raise TypeError("모델 파일의 최상위 구조가 dict가 아닙니다.")

        missing_keys = cls.REQUIRED_BUNDLE_KEYS - set(loaded)
        if missing_keys:
            raise ValueError(f"모델 번들에 필수 항목이 없습니다: {sorted(missing_keys)}")
        return loaded

    def _validate_bundle(self) -> None:
        if not hasattr(self.model, "predict_proba"):
            raise TypeError("저장된 모델에 predict_proba()가 없습니다.")

        if not 0.0 < self.threshold < 1.0:
            raise ValueError(f"저장된 임계값이 올바르지 않습니다: {self.threshold}")

        if not self.feature_cols:
            raise ValueError("저장된 feature_cols가 비어 있습니다.")

        duplicated = pd.Index(self.feature_cols)[pd.Index(self.feature_cols).duplicated()].tolist()
        if duplicated:
            raise ValueError(f"feature_cols에 중복 변수가 있습니다: {duplicated}")

        if len(self.feature_cols) != 53:
            raise ValueError(
                "현재 앱용 모델은 53개 변수를 사용해야 합니다. "
                f"저장된 변수 수: {len(self.feature_cols)}"
            )

        unknown_group_features = [
            feature
            for feature in self.feature_cols
            if not any(
                feature in group_features
                for group_features in CLINICAL_FEATURE_GROUPS.values()
            )
        ]
        if unknown_group_features:
            raise ValueError(
                "임상 입력 단계에 포함되지 않은 모델 변수가 있습니다: "
                f"{unknown_group_features}"
            )

    @staticmethod
    def _is_missing(value: Any) -> bool:
        if value is None or value is pd.NA:
            return True
        if isinstance(value, str):
            return value.strip().lower() in MISSING_TEXT_VALUES
        try:
            missing = pd.isna(value)
        except (TypeError, ValueError):
            return False
        return bool(missing) if np.isscalar(missing) else False

    @staticmethod
    def _normalize_binary_text(feature: str, text: str) -> float:
        lowered = text.strip().lower()
        if lowered in TRUE_TEXT_VALUES:
            return 1.0
        if lowered in FALSE_TEXT_VALUES:
            return 0.0
        raise ValueError(
            f"'{feature}' 값 '{text}'을 0/1로 변환할 수 없습니다. "
            "있음/없음 또는 1/0을 사용하세요."
        )

    @classmethod
    def _normalize_value(cls, feature: str, value: Any) -> float:
        """화면 입력을 모델이 받는 숫자 또는 NaN으로 변환한다."""

        if cls._is_missing(value):
            return np.nan

        if isinstance(value, (bool, np.bool_)):
            return float(value)

        if isinstance(value, str):
            text = value.strip()

            if feature in BINARY_FEATURES:
                return cls._normalize_binary_text(feature, text)

            if feature == "VHD":
                vhd_map = {
                    "n": 0.0,
                    "없음": 0.0,
                    "mild": 1.0,
                    "경도": 1.0,
                    "moderate": 2.0,
                    "중등도": 2.0,
                    "severe": 3.0,
                    "중증": 3.0,
                }
                mapped = vhd_map.get(text.lower())
                if mapped is not None:
                    return mapped

            # Function Class는 원자료의 0/1/2/3 코드를 그대로 받아야 한다.
            # 임상 등급명을 임의로 0~3에 다시 매핑하지 않는다.
            numeric_text = text.replace(",", "")
            if numeric_text.endswith("%"):
                numeric_text = numeric_text[:-1].strip()
            try:
                numeric_value = float(numeric_text)
            except ValueError as error:
                raise ValueError(
                    f"'{feature}' 값 '{value}'을 숫자로 변환할 수 없습니다."
                ) from error
        else:
            try:
                numeric_value = float(value)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"'{feature}' 값 '{value}'을 숫자로 변환할 수 없습니다."
                ) from error

        if not np.isfinite(numeric_value):
            return np.nan

        if feature in BINARY_FEATURES and numeric_value not in (0.0, 1.0):
            raise ValueError(
                f"'{feature}'는 0 또는 1이어야 합니다. 입력값: {numeric_value}"
            )

        return numeric_value

    def prepare_input(
        self,
        patient_data: Mapping[str, Any],
        *,
        reject_unknown_fields: bool = False,
    ) -> tuple[pd.DataFrame, MissingValueReport]:
        """환자 입력을 저장된 53개 변수 순서의 DataFrame으로 만든다."""

        if not isinstance(patient_data, Mapping):
            raise TypeError("patient_data는 변수명: 값 형태의 dict여야 합니다.")

        unknown_fields = sorted(
            str(field) for field in patient_data if field not in self.feature_cols
        )
        if reject_unknown_fields and unknown_fields:
            raise ValueError(f"모델에 없는 입력 항목입니다: {unknown_fields}")

        normalized_row = {
            feature: self._normalize_value(feature, patient_data.get(feature, np.nan))
            for feature in self.feature_cols
        }
        frame = pd.DataFrame([normalized_row], columns=self.feature_cols, dtype=float)
        missing_features = [
            feature for feature in self.feature_cols if pd.isna(frame.at[0, feature])
        ]

        missing_by_group = {
            group: [
                feature
                for feature in self.feature_cols
                if feature in group_features and feature in missing_features
            ]
            for group, group_features in CLINICAL_FEATURE_GROUPS.items()
        }
        missing_by_group_kr = {
            group: [FEATURE_NAME_KR.get(feature, feature) for feature in features]
            for group, features in missing_by_group.items()
        }

        total = len(self.feature_cols)
        entered = total - len(missing_features)
        report = MissingValueReport(
            total_features=total,
            entered_features=entered,
            missing_count=len(missing_features),
            completeness=entered / total,
            missing_features=missing_features,
            missing_features_kr=[
                FEATURE_NAME_KR.get(feature, feature) for feature in missing_features
            ],
            missing_by_group=missing_by_group,
            missing_by_group_kr=missing_by_group_kr,
            unknown_input_fields=unknown_fields,
        )
        return frame, report

    def _positive_class_probability(self, frame: pd.DataFrame) -> float:
        probabilities = np.asarray(self.model.predict_proba(frame), dtype=float)
        if probabilities.shape[0] != 1:
            raise RuntimeError("단일 환자 예측 결과의 행 수가 1이 아닙니다.")

        classes = np.asarray(getattr(self.model, "classes_", [0, 1]))
        positive_indices = np.flatnonzero(classes == 1)
        if len(positive_indices) != 1:
            raise RuntimeError(f"모델의 양성 클래스 1을 찾을 수 없습니다: {classes.tolist()}")

        probability = float(probabilities[0, positive_indices[0]])
        if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise RuntimeError(f"모델이 올바르지 않은 점수를 반환했습니다: {probability}")
        return probability

    def _build_warnings(self, report: MissingValueReport) -> list[str]:
        warnings = []
        if report.missing_count:
            warnings.append(
                f"전체 {report.total_features}개 중 {report.missing_count}개 항목이 "
                "실제 환자값으로 입력되지 않았습니다. 해당 값은 저장된 Pipeline에서 "
                "학습 데이터 중앙값으로 대체됩니다."
            )
        if report.completeness < self.low_completeness_threshold:
            warnings.append(
                f"입력 완성도가 {report.completeness:.1%}로 낮습니다. "
                "결과의 불확실성이 커질 수 있습니다. 미입력은 실제 '없음'과 다릅니다. "
                "확인 가능한 실제 문진·검사값을 추가하면 현재 환자 정보에 근거한 "
                "해석을 더 충실하게 할 수 있습니다."
            )

        if report.unknown_input_fields:
            warnings.append(
                "모델 입력에 포함되지 않아 무시된 항목: "
                + ", ".join(report.unknown_input_fields)
            )

        warnings.append(
            "교육·연구용 의료진 보조 CDSS 프로토타입이며 환자 상태와 전체 "
            "임상정보를 함께 해석해야 합니다."
        )
        return warnings

    def predict(
        self,
        patient_data: Mapping[str, Any],
        *,
        reject_unknown_fields: bool = False,
    ) -> CadPredictionResult:
        """환자 한 명의 CAD 모델 점수와 입력 완성도 정보를 반환한다."""

        frame, missing_report = self.prepare_input(
            patient_data,
            reject_unknown_fields=reject_unknown_fields,
        )
        cad_score = self._positive_class_probability(frame)
        predicted_class = int(cad_score >= self.threshold)
        risk_label = (
            "CAD 고위험 참고군" if predicted_class == 1 else "CAD 저위험 참고군"
        )
        interpretation = (
            "입력된 임상정보가 학습 데이터의 CAD 집단 패턴에 얼마나 가까운지를 "
            "나타내는 모델 점수입니다. 확진 또는 일반 인구의 절대 발병확률이 아닙니다."
        )
        return CadPredictionResult(
            cad_score=cad_score,
            threshold=self.threshold,
            predicted_class=predicted_class,
            risk_label=risk_label,
            model_name=self.model_name,
            feature_group=self.feature_group,
            n_features=len(self.feature_cols),
            missing_report=missing_report,
            warnings=self._build_warnings(missing_report),
            interpretation=interpretation,
        )

    def predict_many(
        self,
        patients: Sequence[Mapping[str, Any]],
        *,
        reject_unknown_fields: bool = False,
    ) -> list[CadPredictionResult]:
        """여러 환자를 같은 검증 규칙으로 순차 예측한다."""

        return [
            self.predict(patient, reject_unknown_fields=reject_unknown_fields)
            for patient in patients
        ]

    def describe(self) -> dict[str, Any]:
        """app.py 시작 시 모델 연결 상태를 확인할 수 있는 정보."""

        return {
            "model_path": str(self.model_path),
            "model_name": self.model_name,
            "feature_group": self.feature_group,
            "n_features": len(self.feature_cols),
            "feature_cols": list(self.feature_cols),
            "threshold": self.threshold,
            "calibration": self.bundle.get("calibration", "unknown"),
            "threshold_source": self.bundle.get("threshold_source"),
            "threshold_selection_rule": self.bundle.get("threshold_selection_rule"),
        }


# =============================================================================
# 3. app.py에서 사용할 간단한 함수
# =============================================================================


@lru_cache(maxsize=4)
def get_cad_model_service(
    model_path: str = str(DEFAULT_MODEL_PATH),
) -> CadModelService:
    """같은 모델 파일을 Streamlit 재실행마다 다시 읽지 않도록 캐시한다."""

    return CadModelService(model_path=model_path)


def predict_cad(
    patient_data: Mapping[str, Any],
    model_path: str | Path = DEFAULT_MODEL_PATH,
    *,
    reject_unknown_fields: bool = False,
) -> dict[str, Any]:
    """app.py에서 바로 호출할 수 있는 함수형 인터페이스."""

    service = get_cad_model_service(str(Path(model_path).expanduser().resolve()))
    result = service.predict(
        patient_data,
        reject_unknown_fields=reject_unknown_fields,
    )
    return result.to_dict()


if __name__ == "__main__":
    service = get_cad_model_service()
    information = service.describe()
    print("CAD 모델 서비스 연결 완료")
    print(f"모델: {information['model_name']}")
    print(f"변수: {information['n_features']}개")
    print("입력 범위: 전체 임상정보")
    print("실제 예측은 app.py에서 predict_cad(patient_data)를 호출하세요.")
