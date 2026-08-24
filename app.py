# app.py
# 실행: streamlit run app.py
# CAD 위험도 분석 CDSS Streamlit 프로토타입
# 기능:
# 1) 임상 변수 입력
# 2) CAD 위험도 예측
# 3) 예측 결과 설명
# 4) EMR 스타일 텍스트 요약
# 5) 의료용어 표준화
# 6) AI 에이전트 상담/해석(규칙 기반)
# 7) 예측 결과 저장/다운로드

import os
import re
import json
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib import font_manager

from cad_model_service import CadModelService, DEFAULT_MODEL_PATH

# 면접용 로컬 실행에서는 프로젝트 폴더의 .env에 저장한 API Key를 자동으로 읽는다.
# 실제 키를 app.py에 직접 적으면 GitHub 등에 노출될 수 있으므로 코드에는 저장하지 않는다.
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().with_name(".env"))
except Exception:
    pass

# 진료지침 기반 RAG(검색증강) 모듈 — 없거나 실패해도 앱은 그대로 동작
try:
    from cad_rag import GuidelineRetriever, build_grounding_block
    RAG_AVAILABLE = True
except Exception:
    RAG_AVAILABLE = False

# 표준 용어 의미 유사도 매칭(문자 n-gram) — cad_rag의 보너스 함수. 없으면 비활성.
try:
    from cad_rag import semantic_match_term
    SEMANTIC_MATCH_AVAILABLE = True
except Exception:
    semantic_match_term = None
    SEMANTIC_MATCH_AVAILABLE = False


# =============================================================================
# 기본 설정
# =============================================================================

st.set_page_config(
    page_title="AI 기반 관상동맥질환 위험도 예측 CDSS",
    page_icon="🫀",
    layout="wide",
)

# Streamlit Community Cloud를 사용할 경우 .streamlit/secrets.toml의 키도 지원한다.
try:
    secret_api_key = str(st.secrets.get("OPENAI_API_KEY", "")).strip()
except Exception:
    secret_api_key = ""

if secret_api_key and not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = secret_api_key

# 탭은 임상 workflow에 맞춰 좌측 → 우측 순서로 진행되도록 안내합니다.
st.markdown(
    """
    <style>
    div[data-testid="stTabs"] [role="tablist"] {
        gap: 0.35rem;
    }
    div[data-testid="stTabs"] button[role="tab"] {
        padding-left: 0.9rem;
        padding-right: 0.9rem;
    }
    .next-step-box {
        text-align: right;
        color: #64748b;
        font-size: 0.92rem;
        margin-top: 1.1rem;
    }

    /* ------------------------------------------------------------------ */
    /* 입력 방식 선택 라디오를 예쁜 세그먼트형(알약) 버튼으로 변환          */
    /* ------------------------------------------------------------------ */
    .st-key-collection_input_mode div[role="radiogroup"] {
        gap: 0.6rem;
        flex-wrap: wrap;
    }
    .st-key-collection_input_mode div[role="radiogroup"] label {
        background: #f8fafc;
        border: 1.5px solid #e2e8f0;
        border-radius: 999px;
        padding: 0.55rem 1.45rem;
        margin: 0;
        cursor: pointer;
        transition: all 0.15s ease;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }
    .st-key-collection_input_mode div[role="radiogroup"] label p {
        font-weight: 600;
        font-size: 0.95rem;
        color: #475569;
        margin: 0;
    }
    .st-key-collection_input_mode div[role="radiogroup"] label:hover {
        border-color: #fb6f70;
        background: #fff5f5;
    }
    /* 기본 라디오 동그라미 숨김 */
    .st-key-collection_input_mode div[role="radiogroup"] label > div:first-child {
        display: none;
    }
    /* 선택된 버튼 강조 */
    .st-key-collection_input_mode div[role="radiogroup"] label:has(input:checked) {
        background: linear-gradient(135deg, #fb6f70 0%, #f4504f 100%);
        border-color: #f4504f;
        box-shadow: 0 4px 10px rgba(244, 80, 79, 0.28);
    }
    .st-key-collection_input_mode div[role="radiogroup"] label:has(input:checked) p {
        color: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

DEFAULT_MODEL_PATHS = [
    str(DEFAULT_MODEL_PATH),
    os.path.join("cad_cdss_final_outputs", "cad_cdss_gradient_boosting_final.joblib"),
    "cad_cdss_gradient_boosting_final.joblib",
]

# 최종 분석 metadata가 있으면 그 값을 우선 사용하고, 없으면 아래 값을 폴백으로 쓴다.
METADATA_PATHS = [
    os.path.join("cad_cdss_final_outputs", "gradient_boosting_final_metadata.json"),
    "gradient_boosting_final_metadata.json",
]

# 사이드바에 표시할 모델 성능 (메타데이터를 못 찾을 때의 폴백 값)
# 최종 모델: GradientBoosting_Raw, 고정 test 셋 1회 평가 기준
MODEL_PERFORMANCE = {
    "AUC": 0.815245,
    "PR-AUC": 0.910585,
    "Recall": 0.860465,
    "Specificity": 0.666667,
    "Precision": 0.860465,
    "F1-score": 0.860465,
    "Accuracy": 0.803279,
    "Brier": 0.158296,
}

IMPORTANCE_FILE_PATHS = [
    os.path.join("cad_cdss_outputs", "feature_importance_permutation.csv"),
    "feature_importance_permutation.csv",
]

VARIABLE_NAME_KR = {
    "Age": "나이", "Sex": "성별", "BMI": "체질량지수",
    "DM": "당뇨병", "HTN": "고혈압",
    "Current Smoker": "현재흡연", "EX-Smoker": "과거흡연",
    "FH": "가족력", "Obesity": "비만", "DLP": "이상지질혈증",
    "CRF": "만성신부전", "CVA": "뇌혈관질환", "CHF": "울혈성심부전",
    "Airway disease": "기도질환", "Thyroid Disease": "갑상선질환",
    "Typical Chest Pain": "전형적흉통", "Atypical": "비전형적흉통",
    "Nonanginal": "비협심증성통증", "Dyspnea": "호흡곤란",
    "Exertional CP": "운동시흉통", "LowTH Ang": "저역치협심증",
    "Function Class": "기능등급", "BP": "혈압", "PR": "맥박수",
    "Edema": "부종", "Weak Peripheral Pulse": "약한말초맥",
    "Lung rales": "폐수포음", "Systolic Murmur": "수축기잡음",
    "Diastolic Murmur": "이완기잡음", "Q Wave": "Q파",
    "St Elevation": "ST상승", "St Depression": "ST하강",
    "Tinversion": "T파역전", "LVH": "좌심실비대",
    "Poor R Progression": "R파진행불량",
    "BBB_LBBB": "좌각차단", "BBB_RBBB": "우각차단",
    "FBS": "공복혈당", "CR": "크레아티닌", "TG": "중성지방",
    "LDL": "저밀도지단백", "HDL": "고밀도지단백",
    "BUN": "혈중요소질소", "ESR": "적혈구침강속도",
    "HB": "헤모글로빈", "K": "칼륨", "Na": "나트륨",
    "WBC": "백혈구", "Lymph": "림프구", "Neut": "호중구",
    "PLT": "혈소판", "EF-TTE": "좌심실박출률",
    "Region RWMA": "국소벽운동이상", "VHD": "판막질환",
    "CAD": "관상동맥질환여부",
}

# 직접 입력 화면은 모델 중요도 순서가 아니라 실제 임상정보 확보 흐름으로 구성한다.
# 문진·과거력·활력징후·신체검사 입력 변수
HISTORY_RISK_FEATURES = [
    "DM", "HTN", "Current Smoker", "EX-Smoker", "FH", "Obesity", "CRF", "CVA",
    "Airway disease", "Thyroid Disease", "CHF", "DLP",
]

SYMPTOM_FEATURES = [
    "Typical Chest Pain", "Atypical", "Nonanginal", "Dyspnea", "LowTH Ang", "Function Class",
]

PHYSICAL_EXAM_FEATURES = [
    "Edema", "Weak Peripheral Pulse", "Lung rales", "Systolic Murmur", "Diastolic Murmur",
]

# 혈액검사 입력 변수
LAB_FEATURES = [
    "FBS", "CR", "TG", "LDL", "HDL", "BUN", "ESR", "HB",
    "K", "Na", "WBC", "Lymph", "Neut", "PLT",
]

# 심전도 입력 변수
ECG_FEATURES = [
    "Q Wave", "St Elevation", "St Depression", "Tinversion",
    "LVH", "Poor R Progression", "BBB_LBBB", "BBB_RBBB",
]

# 심초음파(ECHO) 입력 변수
ECHO_FEATURES = ["EF-TTE", "Region RWMA", "VHD"]

# 모델 입력 순서와는 별개인 화면 표시 순서다. 실제 예측 직전에는
# cad_model_service가 저장된 feature_cols 순서로 다시 정렬한다.
CLINICAL_WORKFLOW_ORDER = [
    "Sex", "Age", "BMI", "BP", "PR",
    *HISTORY_RISK_FEATURES,
    *SYMPTOM_FEATURES,
    *PHYSICAL_EXAM_FEATURES,
    *LAB_FEATURES,
    *ECG_FEATURES,
    *ECHO_FEATURES,
]

BINARY_FEATURES = {
    "Sex", "DM", "HTN", "Current Smoker", "EX-Smoker", "FH", "Obesity", "DLP", "CRF", "CVA", "CHF",
    "Airway disease", "Thyroid Disease", "Typical Chest Pain", "Atypical", "Nonanginal", "Dyspnea",
    "Exertional CP", "LowTH Ang", "Edema", "Weak Peripheral Pulse", "Lung rales", "Systolic Murmur",
    "Diastolic Murmur", "Q Wave", "St Elevation", "St Depression", "Tinversion", "LVH",
    "Poor R Progression", "BBB_LBBB", "BBB_RBBB",
}

ORDINAL_FEATURES = {"Function Class", "Region RWMA", "VHD"}

# 연속형(살아있는 환자에게 0이 임상적으로 불가능한) 변수.
# 이 변수들은 '미입력'이면 0이 아니라 NaN으로 모델에 넣어, 학습 파이프라인의
# 중앙값 대체(median imputation)가 동작하도록 한다. (0을 그대로 넣으면 혈압 0,
# 혈당 0 같은 불가능한 값으로 예측되어 결과가 왜곡됨)
CONTINUOUS_FEATURES = {
    "Age", "BMI", "BP", "PR",
    "FBS", "CR", "TG", "LDL", "HDL", "BUN", "ESR",
    "HB", "K", "Na", "WBC", "Lymph", "Neut", "PLT", "EF-TTE",
}

DEFAULT_VALUES = {
    "Age": 0, "Sex": 1, "BMI": 0,
    "DM": 0, "HTN": 0, "Current Smoker": 0, "EX-Smoker": 0, "FH": 0,
    "Obesity": 0, "DLP": 0, "CRF": 0, "CVA": 0, "CHF": 0, "Airway disease": 0, "Thyroid Disease": 0,
    "Typical Chest Pain": 0, "Atypical": 0, "Nonanginal": 0, "Dyspnea": 0, "Exertional CP": 0, "LowTH Ang": 0,
    "Function Class": 0,
    "BP": 0, "PR": 0, "Edema": 0, "Weak Peripheral Pulse": 0, "Lung rales": 0,
    "Systolic Murmur": 0, "Diastolic Murmur": 0,
    "Q Wave": 0, "St Elevation": 0, "St Depression": 0, "Tinversion": 0, "LVH": 0, "Poor R Progression": 0,
    "BBB_LBBB": 0, "BBB_RBBB": 0,
    "FBS": 0, "CR": 0, "TG": 0, "LDL": 0, "HDL": 0, "BUN": 0,
    "ESR": 0, "HB": 0, "K": 0, "Na": 0, "WBC": 0, "Lymph": 0, "Neut": 0, "PLT": 0,
    "EF-TTE": 0, "Region RWMA": 0, "VHD": 0,
}

TERM_MAP = {
    "흉통": "Typical Chest Pain", "가슴통증": "Typical Chest Pain", "가슴 통증": "Typical Chest Pain",
    "chest pain": "Typical Chest Pain", "cp": "Typical Chest Pain", "전형적 흉통": "Typical Chest Pain",
    "당뇨": "DM", "당뇨병": "DM", "diabetes": "DM", "dm": "DM",
    "고혈압": "HTN", "hypertension": "HTN", "htn": "HTN",
    "흡연": "Current Smoker", "smoker": "Current Smoker", "current smoker": "Current Smoker",
    "과거흡연": "EX-Smoker", "ex smoker": "EX-Smoker", "ex-smoker": "EX-Smoker",
    "가족력": "FH", "family history": "FH", "fh": "FH",
    "이상지질혈증": "DLP", "고지혈증": "DLP", "dlp": "DLP",
    "호흡곤란": "Dyspnea", "숨참": "Dyspnea", "dyspnea": "Dyspnea",
    "t파역전": "Tinversion", "t파 역전": "Tinversion", "t inversion": "Tinversion", "tinversion": "Tinversion",
    "st하강": "St Depression", "st depression": "St Depression",
    "st상승": "St Elevation", "st elevation": "St Elevation",
    "q파": "Q Wave", "q wave": "Q Wave",
    "좌심실박출률": "EF-TTE", "ef": "EF-TTE", "ef-tte": "EF-TTE",
    "국소벽운동이상": "Region RWMA", "rwma": "Region RWMA",
    "중성지방": "TG", "tg": "TG", "triglyceride": "TG",
    "공복혈당": "FBS", "fbs": "FBS",
    "혈압": "BP", "bp": "BP",
    "bmi": "BMI", "체질량지수": "BMI",
}

# =============================================================================
# Clinical NLP / LLM 유틸 함수
# =============================================================================

CAD_NLP_SYSTEM_PROMPT = """
당신은 CAD(관상동맥질환) 위험도 평가를 보조하는 Clinical NLP 어시스턴트입니다.
입력된 EMR 메모, 간호기록, 진료 관련 텍스트를 바탕으로 한국 의무기록 스타일의 SOAP 요약을 작성하세요.

반드시 다음 원칙을 지키세요.
- 진단을 확정하지 말고, 입력된 정보와 CDSS 예측 결과를 분리해서 설명하세요.
- 없는 사실, 약물명, 검사결과를 새로 만들어내지 마세요.
- 출력은 한국어 의무기록 문체(~함/~임/~있음/~없음)를 사용하세요.
- 섹션은 주호소, 현병력, 관련 과거력/위험인자, 검사/소견, CAD CDSS 해석, 평가 및 계획으로 구성하세요.
- 연구/교육용 보조 결과이며 실제 진단을 대체하지 않는다는 문장을 마지막에 포함하세요.
""".strip()


def extract_clinical_features_from_text(text: str, use_semantic: bool = True) -> dict:
    """EMR 자유 텍스트에서 CAD 예측에 필요한 일부 임상 변수를 추출한다.

    정규표현식 + 의료용어 사전 + 간단한 부정표현 처리 기반 Clinical NLP에,
    문자 n-gram 의미 유사도 매칭(use_semantic=True)을 더해 사전에 없는 표현도 보완한다.
    use_semantic=False로 호출하면 정규식/사전 결과만 반환한다(추출 방식 비교용).
    """
    text = text or ""
    text_lower = text.lower()
    extracted = {}

    # 나이: 67세, 67 세, 67-year-old 등
    age_match = re.search(r"(\d{1,3})\s*(?:세|세\s*[남여]성|year[-\s]?old|yo)", text_lower)
    if age_match:
        age = int(age_match.group(1))
        if 0 <= age <= 120:
            extracted["Age"] = age

    # 성별
    if any(word in text_lower for word in ["남성", "남자", "남환", "male", "man"]):
        extracted["Sex"] = 1
    elif any(word in text_lower for word in ["여성", "여자", "여환", "female", "woman"]):
        extracted["Sex"] = 0

    # BMI 직접 입력
    bmi_match = re.search(r"(?:bmi|체질량지수)\s*[:=]?\s*(\d{1,2}(?:\.\d+)?)", text_lower)
    if bmi_match:
        extracted["BMI"] = float(bmi_match.group(1))

    # 혈압: 130/80, BP 130, 혈압 130 등. 모델에는 수축기혈압만 연결
    bp_match = re.search(r"(?:bp|혈압)?\s*(\d{2,3})\s*/\s*(\d{2,3})", text_lower)
    if bp_match:
        extracted["BP"] = float(bp_match.group(1))
    else:
        bp_match = re.search(r"(?:bp|혈압|수축기혈압|sbp)\s*[:=]?\s*(\d{2,3})", text_lower)
        if bp_match:
            extracted["BP"] = float(bp_match.group(1))

    # 심박수
    pr_match = re.search(r"(?:pr|hr|맥박|심박수)\s*[:=]?\s*(\d{2,3})", text_lower)
    if pr_match:
        extracted["PR"] = float(pr_match.group(1))

    # 주요 수치
    numeric_patterns = {
        "FBS": r"(?:fbs|공복혈당)\s*[:=]?\s*(\d{2,3})",
        "TG": r"(?:tg|중성지방|triglyceride)\s*[:=]?\s*(\d{2,4})",
        "LDL": r"(?:ldl)\s*[:=]?\s*(\d{2,4})",
        "HDL": r"(?:hdl)\s*[:=]?\s*(\d{2,4})",
        "CR": r"(?:cr|creatinine|크레아티닌)\s*[:=]?\s*(\d{1,2}(?:\.\d+)?)",
        "EF-TTE": r"(?:ef|ef-tte|좌심실박출률)\s*[:=]?\s*(\d{1,3})\s*%?",
    }
    for feature, pattern in numeric_patterns.items():
        m = re.search(pattern, text_lower)
        if m:
            extracted[feature] = float(m.group(1))

    # 이진 변수: 키워드 + 부정표현 간단 처리
    # (자연어 표현/패러프레이즈도 일부 포함 → 정규식·사전 매칭과 의미 유사도 매칭 모두에 도움)
    positive_terms = {
        "DM": ["당뇨", "당뇨병", "diabetes", "dm"],
        "HTN": ["고혈압", "hypertension", "htn", "혈압이 높", "혈압 높"],
        "Current Smoker": ["현재흡연", "흡연자", "smoker", "current smoker", "흡연", "담배", "흡연력"],
        "EX-Smoker": ["과거흡연", "금연", "ex-smoker", "ex smoker"],
        "FH": ["가족력", "family history", "fh"],
        "DLP": ["이상지질혈증", "고지혈증", "dlp", "dyslipidemia", "콜레스테롤 높"],
        "Typical Chest Pain": ["전형적 흉통", "흉통", "가슴통증", "가슴 통증", "chest pain",
                                 "가슴이 조임", "가슴이 조이", "가슴 압박", "가슴이 답답", "가슴이 아프"],
        "Dyspnea": ["호흡곤란", "숨참", "숨이 차", "숨이 참", "숨가쁨", "dyspnea", "shortness of breath"],
        "Exertional CP": ["운동시흉통", "운동 시 흉통", "운동할 때 흉통", "노작성 흉통", "exertional chest pain"],
        "Tinversion": ["t파 역전", "t파역전", "t inversion", "tinversion"],
        "St Depression": ["st 하강", "st하강", "st depression"],
        "St Elevation": ["st 상승", "st상승", "st elevation"],
        "Q Wave": ["q파", "q wave"],
        "Region RWMA": ["rwma", "국소벽운동이상"],
        "VHD": ["판막질환", "vhd", "valvular heart disease"],
        "Edema": ["부종", "다리가 붓", "발이 붓", "붓는다", "edema"],
    }
    negative_words = ["없음", "없다", "없고", "부인", "아님", "negative", "no ", "denies", "deny"]

    for feature, terms in positive_terms.items():
        for term in terms:
            idx = text_lower.find(term.lower())
            if idx == -1:
                continue
            window = text_lower[max(0, idx - 18): idx + len(term) + 18]
            extracted[feature] = 0 if any(neg in window for neg in negative_words) else 1
            break

    # 의미 유사도 매칭(보완): 사전에 없는 표현(예: "가슴이 조이는 느낌")도
    # 문자 n-gram 유사도로 표준 변수에 매칭해 이진 변수를 보강한다.
    # 정규식/사전으로 이미 잡힌 변수는 건드리지 않고, 부정표현이 있으면 제외한다.
    # 한 구(phrase)를 모든 표준 변수와 독립 비교하는 다중 매칭이라,
    # "가슴이 답답하고 숨이 참"처럼 한 문장에 여러 증상이 있어도 모두 잡는다.
    if use_semantic and SEMANTIC_MATCH_AVAILABLE and semantic_match_term is not None and text.strip():
        try:
            SEM_THRESHOLD = 0.45
            phrases = [p.strip() for p in re.split(r"[.!?。\n,·]", text) if p.strip()]
            for phrase in phrases:
                phrase_lower = phrase.lower()
                if any(neg in phrase_lower for neg in negative_words):
                    continue  # 부정 표현이 든 구는 매칭 보강에서 제외
                for feature, synonyms in positive_terms.items():
                    if feature in extracted:
                        continue  # 정규식/사전 또는 앞 구에서 이미 확정된 변수는 유지
                    matched, score = semantic_match_term(phrase, {feature: synonyms}, min_score=0.0)
                    if matched and score >= SEM_THRESHOLD:
                        extracted[feature] = 1
        except Exception:
            # 유사도 매칭 실패는 무시하고 정규식/사전 결과만 사용
            pass

    return extracted


def make_extracted_features_dataframe(extracted: dict) -> pd.DataFrame:
    rows = []
    for key, value in extracted.items():
        rows.append({
            "추출 변수": key,
            "한글명": VARIABLE_NAME_KR.get(key, key),
            "추출값": value,
            "모델 반영 방식": "이진/범주" if key in BINARY_FEATURES or key in ORDINAL_FEATURES else "수치",
        })
    return pd.DataFrame(rows)


def generate_llm_soap_note(
    clinical_text: str,
    values: dict,
    probability: float,
    threshold: float,
    risk_group: str,
    extracted: dict | None = None,
    model_name: str = "gpt-4o-mini",
    retriever=None,
    reasons: list[str] | None = None,
) -> tuple[str, list[dict]]:
    """OpenAI API로 CAD CDSS용 SOAP 요약을 생성한다. (SOAP초안, 참고근거) 반환.

    retriever가 주어지면 환자의 위험요인으로 진료지침을 검색(RAG)해
    Assessment/Plan의 근거로 함께 사용한다.
    """
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY가 설정되어 있지 않습니다. .env 파일에 OPENAI_API_KEY를 넣어주세요.")

    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError("openai 패키지가 설치되어 있지 않습니다. pip install openai python-dotenv 후 다시 실행하세요.") from e

    extracted = extracted or {}
    compact_values = {
        key: values.get(key)
        for key in ["Age", "Sex", "BMI", "BP", "PR", "DM", "HTN", "DLP", "Current Smoker", "Typical Chest Pain", "Dyspnea", "FBS", "TG", "Tinversion", "EF-TTE", "Region RWMA", "VHD"]
        if key in values
    }
    payload = {
        "free_text": clinical_text,
        "structured_input_values": compact_values,
        "nlp_extracted_values": extracted,
        "cad_cdss_result": {
            "cad_model_score": round(float(probability), 4),
            "threshold": round(float(threshold), 4),
            "risk_group": risk_group,
        },
    }

    # RAG: 위험요인/위험군으로 진료지침 근거 검색
    rag_results: list[dict] = []
    if retriever is not None:
        try:
            query = (risk_group or "") + " 관상동맥질환 평가 및 계획 " + " ".join(reasons or [])
            rag_results = retriever.retrieve(query, k=3)
        except Exception:
            rag_results = []

    user_prompt = (
        "아래 JSON 정보를 바탕으로 CAD CDSS용 Clinical NLP SOAP 요약을 작성하세요.\n"
        "구조화 입력값을 우선으로 사용하고, 자유 텍스트와 구조화 입력값의 차이나 충돌, "
        "누락 여부에 대한 언급(예: '~가 충돌 가능성이 있습니다')은 하지 마세요. 없는 사실은 추정하지 마세요.\n\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )
    if rag_results:
        user_prompt += (
            "\n\n# 참고 진료지침(Assessment/Plan의 일반 의학 판단 근거로 우선 사용)\n"
            + build_grounding_block(rag_results)
            + "\n\n위 진료지침을 참고해 작성했다면 Plan 끝에 (근거: 실제 문서 파일명) 형태로 표기하세요. "
            "'출처명'이라는 단어를 그대로 쓰지 말고, 위 진료지침 블록에 표시된 실제 파일명을 적으세요. "
            "근거에 없는 일반화는 피하세요."
        )

    client = OpenAI()
    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": CAD_NLP_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=1600,
    )
    return resp.choices[0].message.content.strip(), rag_results


def generate_llm_input_summary(
    clinical_text: str,
    values: dict,
    probability: float,
    threshold: float,
    risk_group: str,
    extracted: dict | None = None,
    model_name: str = "gpt-4o-mini",
    retriever=None,
    reasons: list[str] | None = None,
) -> tuple[str, list[dict]]:
    """입력 정보 요약을 RAG+LLM으로 생성한다. (요약문, 참고근거) 반환.

    SOAP 초안과 달리 5~7문장 내외의 짧은 자연어 요약을 만든다.
    retriever가 있으면 진료지침 근거를 함께 참고한다.
    """
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY가 설정되어 있지 않습니다. .env 파일에 OPENAI_API_KEY를 넣어주세요.")

    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError("openai 패키지가 설치되어 있지 않습니다. pip install openai python-dotenv 후 다시 실행하세요.") from e

    extracted = extracted or {}
    compact_values = {
        key: values.get(key)
        for key in ["Age", "Sex", "BMI", "BP", "PR", "DM", "HTN", "DLP", "Current Smoker", "Typical Chest Pain", "Dyspnea", "FBS", "TG", "Tinversion", "EF-TTE", "Region RWMA", "VHD"]
        if key in values
    }
    payload = {
        "free_text": clinical_text,
        "structured_input_values": compact_values,
        "nlp_extracted_values": extracted,
        "cad_cdss_result": {
            "cad_model_score": round(float(probability), 4),
            "threshold": round(float(threshold), 4),
            "risk_group": risk_group,
        },
        "key_risk_factors": reasons or [],
    }

    rag_results: list[dict] = []
    if retriever is not None:
        try:
            query = (risk_group or "") + " 관상동맥질환 위험요인 요약 " + " ".join(reasons or [])
            rag_results = retriever.retrieve(query, k=3)
        except Exception:
            rag_results = []

    system_prompt = (
        "당신은 CAD(관상동맥질환) 위험도 평가를 보조하는 Clinical NLP 어시스턴트입니다. "
        "입력된 환자 정보와 CDSS 결과를 바탕으로 의료진이 한눈에 보는 '입력 정보 요약'을 작성합니다.\n"
        "원칙:\n"
        "- 5~7문장 내외의 간결한 한국어 줄글로 작성하세요. (SOAP 형식·머리글·번호 목록 금지)\n"
        "- 입력된 사실만 사용하고, 없는 약물명·검사결과·진단을 만들어내지 마세요.\n"
        "- 진단을 확정하지 말고, 입력 정보와 CDSS 예측 결과를 분리해서 서술하세요.\n"
        "- 마지막 문장에 연구·교육용 참고 자료라는 점을 짧게 덧붙이세요."
    )
    user_prompt = (
        "아래 JSON 정보를 바탕으로 환자 입력 정보 요약을 작성하세요.\n"
        "구조화 입력값을 우선으로 사용하고, 자유 텍스트와 구조화 입력값의 차이나 충돌, "
        "누락 여부에 대한 언급(예: '~가 충돌 가능성이 있습니다')은 하지 마세요.\n\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )
    if rag_results:
        user_prompt += (
            "\n\n# 참고 진료지침(일반 의학적 맥락 보강용)\n"
            + build_grounding_block(rag_results)
            + "\n\n참고했다면 요약 끝에 (근거: 실제 문서 파일명) 형태로 간단히 표기하세요. "
            "'출처명'이라는 단어를 그대로 쓰지 말고, 위 진료지침 블록에 표시된 실제 파일명을 적으세요. "
            "근거에 없는 일반화는 피하세요."
        )

    client = OpenAI()
    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=700,
    )
    return resp.choices[0].message.content.strip(), rag_results


def transcribe_audio_with_openai(audio_bytes: bytes, file_name: str = "clinical_audio.wav", model_name: str = "whisper-1") -> str:
    """OpenAI Whisper API를 사용해 녹음/오디오 파일을 텍스트로 전사한다."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY가 설정되어 있지 않습니다. .env 파일에 OPENAI_API_KEY를 넣어주세요.")

    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError("openai 패키지가 설치되어 있지 않습니다. pip install openai python-dotenv 후 다시 실행하세요.") from e

    suffix = Path(file_name).suffix or ".wav"
    client = OpenAI()

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        with open(tmp_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model=model_name,
                file=audio_file,
                language="ko",
            )
        return (transcript.text or "").strip()
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

# =============================================================================
# 유틸 함수
# =============================================================================

@st.cache_resource
def load_cad_model_service(model_path: str) -> CadModelService:
    """최종 모델과 입력 검증 규칙을 한 번만 로드한다."""
    return CadModelService(model_path=model_path)


@st.cache_resource(show_spinner=False)
def get_guideline_retriever(use_openai: bool):
    """진료지침 RAG 검색기를 로드(캐시). use_openai가 바뀌면 백엔드를 다시 만든다."""
    if not RAG_AVAILABLE:
        return None
    try:
        retriever = GuidelineRetriever(prefer_openai=use_openai)
        return retriever if retriever.ready else None
    except Exception:
        return None


def find_default_model_path() -> str | None:
    for path in DEFAULT_MODEL_PATHS:
        if os.path.exists(path):
            return path
    return None


def label(feature: str) -> str:
    return f"{VARIABLE_NAME_KR.get(feature, feature)} ({feature})"


def bool_select(feature: str, default: int | None = None) -> int | None:
    """이진형 변수에서 미확인과 실제 '없음(0)'을 구분한다."""
    if feature == "Sex":
        # 원본 데이터 인코딩: 남성=1, 여성=0
        value = st.selectbox(
            label(feature),
            ["미선택", "남성", "여성"],
            index=0,
            key=f"input_{feature}",
        )
        if value == "미선택":
            return None
        return 1 if value == "남성" else 0

    value = st.selectbox(
        label(feature),
        ["미선택", "없음", "있음"],
        index=0,
        key=f"input_{feature}",
    )
    if value == "미선택":
        return None
    return 1 if value == "있음" else 0


def number_input(feature: str, default):
    if feature == "Age":
        return st.number_input(
            label(feature),
            min_value=0,
            max_value=120,
            value=int(default),
            step=1,
            key=f"input_{feature}",
        )

    if feature == "BMI":
        # BMI는 주요 변수 영역에서 키/몸무게를 받아 자동 계산한다.
        # 혹시 다른 위치에서 호출될 경우를 대비해 기본값만 반환한다.
        return float(DEFAULT_VALUES.get("BMI", 25.0))

    if feature == "BP":
        st.markdown("**혈압 / 심박수**")

        col1, col2, col3 = st.columns(3)

        with col1:
            systolic_bp = st.number_input(
                "수축기혈압 SBP (mmHg)",
                min_value=0.0,
                max_value=260.0,
                value=float(default),
                step=1.0,
                key="input_BP_sbp",
            )

        with col2:
            st.number_input(
                "이완기혈압 DBP (mmHg)",
                min_value=0.0,
                max_value=160.0,
                value=0.0,
                step=1.0,
                key="input_BP_dbp",
            )

        with col3:
            pulse_rate = st.number_input(
                "심박수 PR (/min)",
                min_value=0.0,
                max_value=220.0,
                value=float(DEFAULT_VALUES.get("PR", 0)),
                step=1.0,
                key="input_BP_pr",
            )

        st.session_state["calculated_pr"] = pulse_rate

        return systolic_bp

    if feature in {"PR", "FBS", "TG", "LDL", "HDL", "BUN", "ESR", "WBC", "PLT"}:
        return st.number_input(
            label(feature),
            value=float(default),
            step=1.0,
            key=f"input_{feature}",
        )

    return st.number_input(
        label(feature),
        value=float(default),
        step=0.1,
        key=f"input_{feature}",
    )




def collect_manual_values_from_session(feature_cols: list[str]) -> dict:
    """입력 화면을 radio로 전환해도 기존 직접 입력값이 유지되도록 session_state에서 복원한다."""
    restored = {}

    for feature in feature_cols:
        key = f"input_{feature}"
        if feature == "Sex" and key in st.session_state:
            sex_value = st.session_state[key]
            restored[feature] = None if sex_value == "미선택" else (1 if sex_value == "남성" else 0)
        elif feature == "BP" and "input_BP_sbp" in st.session_state:
            restored[feature] = st.session_state.get("input_BP_sbp", DEFAULT_VALUES.get("BP", 0))
        elif feature == "PR" and "input_BP_pr" in st.session_state:
            restored[feature] = st.session_state.get("input_BP_pr", DEFAULT_VALUES.get("PR", 0))
        elif feature == "BMI":
            height_cm = st.session_state.get("input_height_cm_main", 0)
            weight_kg = st.session_state.get("input_weight_kg_main", 0)
            try:
                if float(height_cm) > 0 and float(weight_kg) > 0:
                    restored[feature] = round(float(weight_kg) / ((float(height_cm) / 100) ** 2), 1)
                else:
                    restored[feature] = None
            except Exception:
                restored[feature] = None
        elif feature == "Function Class" and key in st.session_state:
            label_value = st.session_state[key]
            if label_value == "미선택":
                restored[feature] = None
            elif "Class I:" in label_value:
                restored[feature] = 0
            elif "Class II:" in label_value:
                restored[feature] = 1
            elif "Class III:" in label_value:
                restored[feature] = 2
            elif "Class IV:" in label_value:
                restored[feature] = 3
        elif feature == "Region RWMA" and key in st.session_state:
            label_value = st.session_state[key]
            if label_value == "미선택":
                restored[feature] = None
            elif "RWMA 없음" in label_value:
                restored[feature] = 0
            elif "1개" in label_value:
                restored[feature] = 1
            elif "2개" in label_value:
                restored[feature] = 2
            elif "3개" in label_value:
                restored[feature] = 3
            elif "4개" in label_value:
                restored[feature] = 4
        elif feature == "VHD" and key in st.session_state:
            label_value = st.session_state[key]
            if label_value == "미선택":
                restored[feature] = None
            elif "없음" in label_value:
                restored[feature] = 0
            elif "경도" in label_value:
                restored[feature] = 1
            elif "중등도" in label_value:
                restored[feature] = 2
            elif "중증" in label_value:
                restored[feature] = 3
        elif feature in BINARY_FEATURES and key in st.session_state:
            binary_value = st.session_state[key]
            restored[feature] = None if binary_value == "미선택" else (1 if binary_value == "있음" else 0)
        elif key in st.session_state:
            restored[feature] = st.session_state[key]

    return restored

def feature_input_widget(feature: str):
    default = DEFAULT_VALUES.get(feature, 0)
    
    if feature == "PR" and "calculated_pr" in st.session_state:
        return st.session_state["calculated_pr"]
    
    if feature in BINARY_FEATURES:
        return bool_select(feature, int(default))

    if feature == "Function Class":
        function_class_options = {
            "미선택": None,
            "정상 / Class I: 일상 활동 시 숨참·피로·흉통 없음": 0,
            "경증 / Class II: 일반 활동 시 숨참·피로·흉통 발생": 1,
            "중등도 / Class III: 가벼운 활동에도 숨참·피로·흉통 발생": 2,
            "중증 / Class IV: 안정 시에도 증상 또는 활동 불가": 3,
        }

        function_labels = list(function_class_options.keys())
        selected_label = st.selectbox(
            "기능등급 (Function Class)",
            options=function_labels,
            index=0,
            key=f"input_{feature}",
            help=(
                "Function Class는 신체활동 시 숨참, 피로, 흉통 등 증상과 "
                "활동 제한 정도를 나타내는 변수입니다. "
                "본 앱에서는 원 데이터셋 기준으로 0=Class I, 1=Class II, "
                "2=Class III, 3=Class IV로 모델에 입력합니다."
            ),
        )

        return function_class_options[selected_label]

    if feature == "Region RWMA":
        rwma_options = {
            "미선택": None,
            "RWMA 없음": 0,
            "1개 영역에서 RWMA 관찰": 1,
            "2개 영역에서 RWMA 관찰": 2,
            "3개 영역에서 RWMA 관찰": 3,
            "4개 영역 이상에서 RWMA 관찰": 4,
        }

        rwma_labels = list(rwma_options.keys())
        selected_label = st.selectbox(
            "국소벽운동이상 관찰 영역 수 (Region RWMA)",
            options=rwma_labels,
            index=0,
            key=f"input_{feature}",
            help=(
                "Region RWMA는 심초음파에서 국소벽운동이상이 관찰된 영역 수를 의미합니다. "
                "RWMA 없음은 벽운동장애가 관찰되지 않은 상태이며, "
                "1~4는 벽운동장애가 관찰된 영역 수 증가를 나타냅니다. "
                "중증도 점수가 아니라 관찰 영역 수 기준입니다."
            ),
        )

        return rwma_options[selected_label]

    if feature == "VHD":
        vhd_options = {
            "미선택": None,
            "없음 (N)": 0,
            "경도 (mild)": 1,
            "중등도 (Moderate)": 2,
            "중증 (Severe)": 3,
        }

        vhd_labels = list(vhd_options.keys())
        selected_label = st.selectbox(
            "판막질환 정도 (VHD)",
            options=vhd_labels,
            index=0,
            key=f"input_{feature}",
            help=(
                "VHD는 판막질환 정도를 의미합니다. "
                "원본 데이터의 N/mild/Moderate/Severe 값을 "
                "없음/경도/중등도/중증으로 표시합니다. "
                "모델에는 0, 1, 2, 3 값으로 입력됩니다."
            ),
        )

        return vhd_options[selected_label]

    return number_input(feature, default)


def render_feature_grid(
    features: list[str],
    values: dict,
    feature_cols: list[str],
    columns: int = 3,
) -> None:
    """지정한 임상 변수들을 같은 간격의 입력 그리드로 표시한다."""
    available_features = [feature for feature in features if feature in feature_cols]
    if not available_features:
        return

    grid_columns = st.columns(columns)
    for index, feature in enumerate(available_features):
        with grid_columns[index % columns]:
            values[feature] = feature_input_widget(feature)

def find_existing_file(paths: list[str]) -> str | None:
    for path in paths:
        if os.path.exists(path):
            return path
    return None


def load_permutation_importance() -> pd.DataFrame:
    """학습/평가 단계에서 저장한 permutation importance 파일을 불러온다."""
    importance_path = find_existing_file(IMPORTANCE_FILE_PATHS)
    if not importance_path:
        return pd.DataFrame()

    df = pd.read_csv(importance_path)

    # 다양한 컬럼명에 대응
    col_map = {}
    for col in df.columns:
        lower = col.lower()
        if lower in ["feature", "variable", "변수"]:
            col_map[col] = "변수"
        elif lower in ["importance", "importance_mean", "mean_importance", "permutation_importance", "auc_importance"]:
            col_map[col] = "중요도"
        elif lower in ["importance_std", "std", "std_importance"]:
            col_map[col] = "표준편차"

    df = df.rename(columns=col_map)

    if "변수" not in df.columns:
        df = df.rename(columns={df.columns[0]: "변수"})

    if "중요도" not in df.columns:
        numeric_cols = [c for c in df.columns if c != "변수" and pd.api.types.is_numeric_dtype(df[c])]
        if numeric_cols:
            df = df.rename(columns={numeric_cols[0]: "중요도"})
        else:
            return pd.DataFrame()

    df["한글명"] = df["변수"].map(lambda x: VARIABLE_NAME_KR.get(str(x), str(x)))
    df["중요도"] = pd.to_numeric(df["중요도"], errors="coerce").fillna(0)
    df = df.sort_values("중요도", ascending=False)
    return df


def get_top_global_importance(top_n: int = 10) -> pd.DataFrame:
    """Permutation importance 기반 전체 모델 설명 결과를 반환한다."""
    df = load_permutation_importance()
    if df.empty:
        return pd.DataFrame()
    cols = ["한글명", "변수", "중요도"]
    if "표준편차" in df.columns:
        cols.append("표준편차")
    return df.head(top_n)[cols]


def load_model_performance() -> dict:
    """최종 metadata의 test_result를 사용하고, 없으면 폴백 값을 쓴다.

    하드코딩된 수치가 재학습 후 실제 모델과 어긋나는 문제를 막는다.
    """
    path = find_existing_file(METADATA_PATHS)
    if not path:
        return MODEL_PERFORMANCE
    try:
        with open(path, encoding="utf-8") as f:
            meta = json.load(f)
        fr = meta.get("test_result", {}) or meta.get("final_result", {}) or {}
        mapped = {
            "AUC": fr.get("roc_auc", fr.get("auc")),
            "PR-AUC": fr.get("pr_auc"),
            "Recall": fr.get("recall"),
            "Specificity": fr.get("specificity"),
            "Precision": fr.get("precision"),
            "F1-score": fr.get("f1"),
            "Accuracy": fr.get("accuracy"),
            "Brier": fr.get("brier"),
        }
        return {k: (v if v is not None else MODEL_PERFORMANCE.get(k)) for k, v in mapped.items()}
    except Exception:
        return MODEL_PERFORMANCE


def render_model_performance_sidebar() -> None:
    st.sidebar.markdown("### 최종 테스트 세트 성능")
    performance = load_model_performance()
    perf_cols = [
        ("AUC", "AUC"),
        ("PR-AUC", "PR-AUC"),
        ("Recall", "Recall"),
        ("Specificity", "Specificity"),
        ("F1-score", "F1"),
        ("Precision", "Precision"),
        ("Accuracy", "Accuracy"),
        ("Brier", "Brier"),
    ]
    for key, label_text in perf_cols:
        value = performance.get(key)
        if value is not None:
            st.sidebar.write(f"{label_text}: `{value:.3f}`")


def has_meaningful_input(values: dict, feature_cols: list[str]) -> bool:
    """사용자가 실제로 입력한 값이 있는지 확인한다.

    연속형의 0은 아직 입력하지 않은 화면 기본값으로 보지만, 이진형·서열형의
    0은 의료진이 명시적으로 선택한 '없음/정상'이므로 유효한 입력으로 본다.
    """
    for key in feature_cols:
        value = values.get(key)
        if value is None or value == "":
            continue
        if key in CONTINUOUS_FEATURES:
            try:
                if np.isfinite(float(value)) and float(value) != 0:
                    return True
            except (TypeError, ValueError):
                continue
        else:
            try:
                if not pd.isna(value):
                    return True
            except (TypeError, ValueError):
                return True
    return False


def make_input_dataframe(values: dict, feature_cols: list[str]) -> pd.DataFrame:
    row = {}
    for col in feature_cols:
        row[col] = values.get(col)
    return pd.DataFrame([row], columns=feature_cols)


def prepare_values_for_service(values: dict, feature_cols: list[str]) -> dict:
    """화면의 미입력 표시를 서비스가 이해하는 결측값으로 변환한다.

    연속형 입력창의 0은 화면상 미입력 상태다. 이진형·서열형의 명시적 0은
    실제 '없음/정상' 코드이므로 그대로 보존한다.
    """
    prepared = {}
    for col in feature_cols:
        value = values.get(col)
        if col in CONTINUOUS_FEATURES:
            try:
                if value in (None, "", "0") or float(value) == 0:
                    value = None
            except (TypeError, ValueError):
                value = None
        prepared[col] = value
    return prepared


def get_category_text(feature: str, value) -> str:
    try:
        value = float(value)
    except Exception:
        return ""

    if feature == "Age":
        if value < 45:
            return "<45세"
        if value < 60:
            return "45-59세"
        if value < 75:
            return "60-74세"
        return "75세 이상"
    if feature == "BMI":
        if value < 18.5:
            return "저체중"
        if value < 25:
            return "정상/과체중"
        if value < 30:
            return "비만1"
        return "비만2+"
    if feature == "BP":
        if value < 120:
            return "정상범위"
        if value < 140:
            return "주의"
        return "고혈압범위"
    if feature == "FBS":
        if value < 100:
            return "정상"
        if value < 126:
            return "공복혈당장애"
        return "당뇨범위"
    return ""


def extract_standard_terms(text: str) -> pd.DataFrame:
    text_lower = text.lower()
    rows = []
    used = set()
    for raw_term, std_var in TERM_MAP.items():
        if raw_term.lower() in text_lower and (raw_term, std_var) not in used:
            used.add((raw_term, std_var))
            rows.append({
                "입력 표현": raw_term,
                "표준 변수": std_var,
                "표준 변수명": VARIABLE_NAME_KR.get(std_var, std_var),
                "권장 처리": "언급됨/양성 후보" if std_var in BINARY_FEATURES else "수치 확인 필요",
            })
    return pd.DataFrame(rows)


def _has_final_consonant(word: str) -> bool:
    """단어의 마지막 글자에 받침이 있는지 판단한다. (조사 자동 선택용)

    한글 음절은 유니코드로 받침 유무를 계산하고,
    숫자/기호로 끝나는 경우는 한국어 발음 기준으로 받침 여부를 추정한다.
    """
    if not word:
        return False
    last = word[-1]
    code = ord(last)
    # 한글 음절 영역: (코드 - 0xAC00) % 28 == 0 이면 받침 없음
    if 0xAC00 <= code <= 0xD7A3:
        return (code - 0xAC00) % 28 != 0
    # 숫자 끝소리 받침: 0(영)·1(일)·3(삼)·6(육)·7(칠)·8(팔)
    if last.isdigit():
        return last in "013678"
    return False


def _with_josa(word: str, with_batchim: str, without_batchim: str) -> str:
    """받침 유무에 맞는 조사를 단어 뒤에 붙여 반환한다. 예: _with_josa('고혈압', '이', '가')"""
    return word + (with_batchim if _has_final_consonant(word) else without_batchim)


def _safe_float(value, default: float = 0.0) -> float:
    """None·빈 문자열·NaN을 안전하게 기본값으로 변환한다."""
    try:
        if value is None or value is pd.NA:
            return float(default)
        if isinstance(value, str) and not value.strip():
            return float(default)
        numeric = float(value)
        return numeric if np.isfinite(numeric) else float(default)
    except (TypeError, ValueError, OverflowError):
        return float(default)


def _safe_int(value, default: int = 0) -> int:
    """이진형·서열형 값의 None을 안전하게 정수 기본값으로 변환한다."""
    return int(_safe_float(value, float(default)))


def make_emr_summary(values: dict, probability: float, threshold: float, risk_group: str) -> str:
    """입력된 값과 예측 결과를 바탕으로 화면용 요약문을 만든다.

    0은 기본 미입력값으로 볼 수 있으므로, 실제 입력된 값만 문장에 포함한다.
    """
    sex_value = _safe_float(values.get("Sex"), np.nan)
    sex_text = "남성" if sex_value == 1 else ("여성" if sex_value == 0 else "")
    age = _safe_int(values.get("Age"))

    parts = []

    # 기본값이 모두 0일 때 "여성 환자입니다."가 자동으로 나오지 않게 처리합니다.
    # 나이가 입력된 경우에만 환자 기본정보 문장을 생성합니다.
    if age > 0:
        demographic_text = f"{age}세"
        if sex_text:
            demographic_text += f" {sex_text}"
        parts.append(f"{demographic_text} 환자입니다.")

    histories = []
    for col, name in [
        ("DM", "당뇨병"),
        ("HTN", "고혈압"),
        ("DLP", "이상지질혈증"),
        ("Current Smoker", "현재 흡연"),
        ("FH", "가족력"),
    ]:
        if _safe_int(values.get(col)) == 1:
            histories.append(name)
    if histories:
        joined = ", ".join(histories)
        parts.append(f"과거력 및 위험인자로 {_with_josa(joined, '이', '가')} 있습니다.")

    symptoms = []
    for col, name in [
        ("Typical Chest Pain", "전형적 흉통"),
        ("Atypical", "비전형적 흉통"),
        ("Nonanginal", "비협심증성 통증"),
        ("Dyspnea", "호흡곤란"),
    ]:
        if _safe_int(values.get(col)) == 1:
            symptoms.append(name)
    if symptoms:
        joined = ", ".join(symptoms)
        parts.append(f"증상으로는 {_with_josa(joined, '이', '가')} 확인됩니다.")

    value_items = []

    bmi = _safe_float(values.get("BMI"))
    if bmi > 0:
        value_items.append(f"BMI {bmi:.1f}({get_category_text('BMI', bmi)})")

    bp = _safe_float(values.get("BP"))
    if bp > 0:
        value_items.append(f"혈압 {bp:.0f}mmHg({get_category_text('BP', bp)})")

    pr = _safe_float(values.get("PR"))
    if pr > 0:
        value_items.append(f"심박수 {pr:.0f}/min")

    fbs = _safe_float(values.get("FBS"))
    if fbs > 0:
        value_items.append(f"공복혈당 {fbs:.0f}mg/dL({get_category_text('FBS', fbs)})")

    tg = _safe_float(values.get("TG"))
    if tg > 0:
        value_items.append(f"중성지방 {tg:.0f}")

    ef = _safe_float(values.get("EF-TTE"))
    if ef > 0:
        value_items.append(f"EF-TTE {ef:.1f}%")

    if value_items:
        parts.append("주요 검사 수치는 " + ", ".join(value_items) + " 등입니다.")

    ecg = []
    for col, name in [
        ("Tinversion", "T파 역전"),
        ("St Depression", "ST 하강"),
        ("St Elevation", "ST 상승"),
        ("Q Wave", "Q파 이상"),
    ]:
        if _safe_int(values.get(col)) == 1:
            ecg.append(name)
    if ecg:
        joined = ", ".join(ecg)
        parts.append(f"심전도에서 {_with_josa(joined, '이', '가')} 관찰됩니다.")

    if _safe_float(values.get("Region RWMA")) > 0:
        parts.append("심초음파에서 국소벽운동이상 소견이 확인됩니다.")

    # 화면 상단의 예측 탭에서 이미 확률과 분류를 보여주므로,
    # 자동 결과 요약에서는 확률/threshold 문장을 반복하지 않는다.
    parts.append("본 요약은 입력된 임상정보를 바탕으로 작성된 연구·교육용 참고 문장입니다.")

    return "\n".join(parts)



def make_reason_list(values: dict) -> list[str]:
    reasons = []
    checks = [
        ("Typical Chest Pain", 1, "전형적 흉통 있음"),
        ("DM", 1, "당뇨병 병력"),
        ("HTN", 1, "고혈압 병력"),
        ("DLP", 1, "이상지질혈증"),
        ("Current Smoker", 1, "현재 흡연"),
        ("FH", 1, "가족력"),
        ("Dyspnea", 1, "호흡곤란"),
        ("Tinversion", 1, "T파 역전"),
        ("St Depression", 1, "ST 하강"),
        ("Q Wave", 1, "Q파 이상"),
    ]
    for col, expected, text in checks:
        if _safe_int(values.get(col)) == expected:
            reasons.append(text)
    if _safe_float(values.get("BP")) >= 140:
        reasons.append("혈압 고혈압범위")
    if _safe_float(values.get("FBS")) >= 126:
        reasons.append("공복혈당 당뇨범위")
    if _safe_float(values.get("BMI")) >= 30:
        reasons.append("BMI 비만범위")
    # EF-TTE가 0이면 아직 입력하지 않은 값으로 보고 위험요인에 표시하지 않습니다.
    # EF-TTE를 실제로 입력했고 50 미만일 때만 좌심실박출률 저하로 표시합니다.
    ef_value = _safe_float(values.get("EF-TTE"))
    if 0 < ef_value < 50:
        reasons.append("좌심실박출률 저하")
    if _safe_float(values.get("Region RWMA")) > 0:
        reasons.append("국소벽운동이상")
    return list(dict.fromkeys(reasons))


def get_risk_level(cad_score: float, decision_threshold: float) -> tuple[str, str]:
    """모델 점수를 저장된 최종 임계값 기준의 참고군으로 변환한다.

    색상 기준:
    - 0.30 미만: 낮은 참고군
    - 0.30 이상, 최종 임계값 미만: 주의 참고군
    - 최종 임계값 이상: 고위험 참고군
    """
    if cad_score < 0.30:
        return "CAD 낮은 참고군", "#16a34a"
    if cad_score < decision_threshold:
        return "CAD 주의 참고군", "#f59e0b"
    return "CAD 고위험 참고군", "#ef4444"


def render_risk_badge(cad_score: float, decision_threshold: float) -> None:
    risk_label, risk_color = get_risk_level(cad_score, decision_threshold)
    st.markdown(
        f"""
        <div style="
            border-left: 8px solid {risk_color};
            background: rgba(248, 250, 252, 0.95);
            padding: 18px 22px;
            border-radius: 12px;
            margin: 12px 0 18px 0;
        ">
            <div style="font-size: 0.95rem; color: #64748b; margin-bottom: 6px;">
                CAD 위험도 분류
            </div>
            <div style="font-size: 2rem; font-weight: 800; color: {risk_color};">
                {risk_label}
            </div>
            <div style="font-size: 1rem; color: #334155; margin-top: 6px;">
                모델 점수 {cad_score:.1%}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def set_korean_matplotlib_font() -> None:
    """한글이 깨지지 않도록 사용 가능한 폰트를 우선 적용한다."""
    preferred_fonts = ["Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR", "DejaVu Sans"]
    available_fonts = {f.name for f in font_manager.fontManager.ttflist}

    for font_name in preferred_fonts:
        if font_name in available_fonts:
            plt.rcParams["font.family"] = font_name
            break

    plt.rcParams["axes.unicode_minus"] = False



def render_xai_bar_chart(xai_df: pd.DataFrame, top_n: int = 5) -> None:
    """예측 근거를 보기 쉬운 작은 가로 막대그래프로 표시한다."""
    if xai_df.empty or "절대영향도" not in xai_df.columns:
        return

    set_korean_matplotlib_font()

    chart_df = xai_df.head(top_n).copy()
    chart_df["절대영향도"] = pd.to_numeric(chart_df["절대영향도"], errors="coerce").fillna(0)
    chart_df = chart_df.sort_values("절대영향도", ascending=True)

    fig, ax = plt.subplots(figsize=(5.8, 2.8))
    y_labels = chart_df["변수"] if "변수" in chart_df.columns else chart_df["한글명"]
    bars = ax.barh(y_labels, chart_df["절대영향도"], height=0.48, color="#475569")
    ax.set_title("Top 5 Feature Impact", fontsize=12, pad=8)
    ax.set_xlabel("Impact score", fontsize=9)
    ax.set_ylabel("")
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", labelsize=8)
    ax.tick_params(axis="x", labelsize=8)

    max_val = float(chart_df["절대영향도"].max()) if len(chart_df) else 0.0
    offset = max(max_val * 0.03, 0.0015)
    ax.set_xlim(0, max_val * 1.18 if max_val > 0 else 1)

    for bar, value in zip(bars, chart_df["절대영향도"]):
        ax.text(
            bar.get_width() + offset,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.4f}",
            va="center",
            ha="left",
            fontsize=8,
        )

    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)


def get_feature_importance_from_model(model):
    """Pipeline/RandomForest 등에서 변수 중요도를 최대한 안전하게 가져온다."""
    candidates = [model]

    if hasattr(model, "named_steps"):
        candidates.extend(list(model.named_steps.values()))

    if hasattr(model, "steps"):
        candidates.extend([step_model for _, step_model in model.steps])

    for candidate in candidates:
        if hasattr(candidate, "feature_importances_"):
            return candidate.feature_importances_

    for candidate in candidates:
        if hasattr(candidate, "coef_"):
            coef = candidate.coef_
            try:
                return np.abs(coef[0])
            except Exception:
                return np.abs(coef).ravel()

    return None


def explain_prediction(
    model,
    input_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, str | None]:
    """개별 SHAP 설명을 만들고, 실패하면 전체 모델 중요도로 안전하게 대체한다."""

    estimator = model
    preprocessor = None
    if hasattr(model, "named_steps"):
        estimator = model.named_steps.get("model", model)
        preprocessor = model.named_steps.get("preprocess")

    try:
        import shap

        if not hasattr(estimator, "feature_importances_"):
            raise RuntimeError("현재 모델은 SHAP TreeExplainer 적용 대상이 아닙니다.")

        X = input_df[feature_cols]
        X_input = preprocessor.transform(X) if preprocessor is not None else X.to_numpy()
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(X_input)

        # SHAP 버전에 따라 반환 형태가 다르므로 양성 클래스(CAD=1)의 값을 추출한다.
        if isinstance(shap_values, list):
            class_vals = np.asarray(shap_values[-1])[0]
        else:
            arr = np.asarray(shap_values)
            if arr.ndim == 3:
                class_vals = arr[0, :, -1]
            elif arr.ndim == 2:
                class_vals = arr[0]
            else:
                raise RuntimeError(f"예상하지 못한 SHAP 출력 형태입니다: {arr.shape}")

        class_vals = np.asarray(class_vals, dtype=float).ravel()
        if len(class_vals) != len(feature_cols):
            raise RuntimeError(
                f"SHAP 변수 수({len(class_vals)})와 모델 입력 변수 수({len(feature_cols)})가 다릅니다."
            )

        xai_df = pd.DataFrame({
            "한글명": [VARIABLE_NAME_KR.get(f, f) for f in feature_cols],
            "변수": feature_cols,
            "입력값": [input_df.iloc[0][f] for f in feature_cols],
            "영향도": class_vals,
        })
        xai_df["영향 방향"] = xai_df["영향도"].apply(
            lambda x: "위험도 증가 방향" if x > 0 else "위험도 감소 방향"
        )
        xai_df["절대영향도"] = xai_df["영향도"].abs()
        xai_df["설명 방식"] = "SHAP"
        return xai_df.sort_values("절대영향도", ascending=False), None

    except Exception as shap_error:
        error_message = f"{type(shap_error).__name__}: {shap_error}"

        # shap이 설치되지 않았거나 개별 계산이 실패해도 모델 자체 중요도를 표시한다.
        model_importance = get_feature_importance_from_model(model)
        if model_importance is not None:
            importance = np.asarray(model_importance, dtype=float).ravel()
            if len(importance) == len(feature_cols):
                fallback_df = pd.DataFrame({
                    "한글명": [VARIABLE_NAME_KR.get(f, f) for f in feature_cols],
                    "변수": feature_cols,
                    "입력값": [input_df.iloc[0][f] for f in feature_cols],
                    "영향도": importance,
                })
                fallback_df["절대영향도"] = fallback_df["영향도"].abs()
                fallback_df["영향 방향"] = "전체 모델 중요도"
                fallback_df["설명 방식"] = "Gradient Boosting feature_importances_"
                return fallback_df.sort_values("절대영향도", ascending=False), error_message

        global_df = get_top_global_importance(top_n=15)
        if global_df.empty:
            return pd.DataFrame(), error_message

        global_df = global_df.copy()
        global_df["입력값"] = global_df["변수"].map(
            lambda f: input_df.iloc[0][f] if f in input_df.columns else ""
        )
        global_df["영향도"] = global_df["중요도"]
        global_df["절대영향도"] = global_df["중요도"].abs()
        global_df["영향 방향"] = "전체 모델 중요도"
        global_df["설명 방식"] = "Permutation importance"
        return global_df.sort_values("절대영향도", ascending=False), error_message


def agent_answer(question: str, probability: float, threshold: float, risk_group: str, reasons: list[str], emr_summary: str) -> str:
    q = question.lower().strip()
    is_high_risk = probability >= threshold
    risk_text = "고위험 참고군" if is_high_risk else "저위험 참고군"
    reason_text = ", ".join(reasons[:6]) if reasons else "현재 입력값에서 뚜렷한 위험요인은 제한적입니다"

    if not q:
        return "질문을 입력하면 현재 CAD 예측 결과를 짧게 해석해 드립니다."

    if any(k in q for k in ["고위험", "높은 위험", "위험이 높", "왜", "이유", "위험", "해석", "결과", "설명"]):
        return (
            f"현재 CAD 모델 점수는 {probability:.1%}이고, 설정된 기준값 {threshold:.2f}와 비교하면 **{risk_text}**입니다. "
            f"이 판단에 참고할 수 있는 입력 항목은 {reason_text}입니다. "
            "즉, 이 결과는 확정 진단이 아니라 추가 검사나 의료진 판단을 도와주는 선별 참고값으로 보는 것이 좋습니다."
        )

    if any(k in q for k in ["저위험", "낮은 위험", "위험이 낮"]):
        return (
            f"현재 CAD 모델 점수는 {probability:.1%}로, 기준값 {threshold:.2f} 대비 **{risk_text}**입니다. "
            f"참고 요인은 {reason_text}입니다. "
            "증상이나 심전도/심초음파 소견이 추가되면 예측 결과가 달라질 수 있습니다."
        )

    if any(k in q for k in ["emr", "기록", "차트", "soap", "요약"]):
        return emr_summary

    if any(k in q for k in ["추가", "검사", "확인"]):
        return (
            "추가로 확인하면 좋은 항목은 흉통 양상, 운동 시 증상 악화 여부, 심전도 변화, "
            "심초음파 EF/RWMA, 공복혈당·지질수치, 당뇨·고혈압·흡연력입니다. "
            f"현재 입력 기준 참고 요인은 {reason_text}입니다."
        )

    if any(k in q for k in ["주의", "한계", "진단"]):
        return (
            "이 앱은 연구·교육용 CAD CDSS 프로토타입입니다. "
            "예측값은 CAD 가능성을 선별해 보는 참고 결과이며, 실제 진단은 의료진 판단과 CAG/검사 결과를 함께 봐야 합니다."
        )

    return (
        f"현재 CAD 모델 점수는 {probability:.1%}, 분류는 **{risk_text}**입니다. "
        f"주요 참고 요인은 {reason_text}입니다."
    )


def generate_cdss_chat_answer(
    user_question: str,
    values: dict,
    probability: float,
    threshold: float,
    risk_group: str,
    reasons: list[str],
    emr_summary: str,
    use_llm: bool = True,
    model_name: str = "gpt-4o-mini",
    retriever=None,
) -> tuple[str, list[dict]]:
    """CDSS 결과에 대해 챗봇 형태로 답변한다. (답변, 참고근거 목록) 반환.

    OpenAI API Key가 있으면 LLM 기반으로 답변하고,
    없거나 실패하면 규칙 기반 답변(agent_answer)으로 대체한다.
    retriever가 주어지면 진료지침에서 근거를 검색(RAG)해 함께 사용한다.
    """
    user_question = (user_question or "").strip()
    if not user_question:
        return "질문을 입력해 주세요.", []

    # RAG: 질문 + 주요 위험요인으로 진료지침 근거 검색
    rag_results: list[dict] = []
    if retriever is not None:
        try:
            query = user_question + " " + " ".join(reasons or [])
            rag_results = retriever.retrieve(query, k=3)
        except Exception:
            rag_results = []

    fallback_answer = agent_answer(user_question, probability, threshold, risk_group, reasons, emr_summary)
    if rag_results:
        top = rag_results[0]
        fallback_answer += f"\n\n📚 참고 진료지침({top['source']}): {top['text']}"

    if not use_llm or not os.getenv("OPENAI_API_KEY"):
        return fallback_answer, rag_results

    try:
        from openai import OpenAI

        compact_values = {
            key: values.get(key)
            for key in [
                "Age", "Sex", "BMI", "BP", "PR",
                "DM", "HTN", "DLP", "Current Smoker", "EX-Smoker", "FH",
                "Typical Chest Pain", "Atypical", "Nonanginal", "Dyspnea", "Exertional CP",
                "FBS", "TG", "LDL", "HDL", "Tinversion", "St Depression", "St Elevation",
                "EF-TTE", "Region RWMA", "VHD",
            ]
            if key in values
        }

        payload = {
            "user_question": user_question,
            "cad_cdss_result": {
                "cad_model_score": round(float(probability), 4),
                "threshold": round(float(threshold), 4),
                "risk_group": risk_group,
            },
            "major_reference_factors": reasons,
            "input_values": compact_values,
            "basic_summary": emr_summary,
        }

        system_prompt = """
당신은 관상동맥질환(CAD) 위험도 예측 CDSS의 해석 보조 챗봇입니다.
반드시 아래 원칙을 지키세요.

- 한국어로 답변하세요.
- 진단을 확정하지 마세요.
- 입력값과 예측 결과를 바탕으로 설명하되, 없는 사실을 만들지 마세요.
- '참고 진료지침' 섹션이 주어지면, 일반 의학 설명은 그 내용을 우선 근거로 삼고,
  사용한 근거를 답변 끝에 (근거: 실제 문서 파일명) 형태로 표시하세요.
  '출처명'이라는 단어를 그대로 쓰지 말고, 참고 진료지침 블록에 표시된 실제 파일명을 적으세요.
  참고한 근거가 없으면 (근거: ...) 표기 자체를 생략하세요.
- 사용자가 위험요인을 물으면 현재 입력된 정보에서 확인 가능한 항목만 말하세요.
- 사용자가 추가 확인사항을 물으면 흉통 양상, 심전도, 심초음파, 혈당/지질, 과거력 등을 일반적인 확인 항목으로 안내하세요.
- 답변은 면접 시연용처럼 짧고 명확하게 3~5문장으로 작성하세요.
- 먼저 CAD 모델 점수와 참고군을 말하고, 그 다음 현재 입력된 위험요인을 근거로 설명하세요.
- 사용자가 이해하기 쉬운 임상 표현을 쓰고, 과장된 진단 표현은 피하세요.
- 마지막에 매번 긴 면책문구를 반복하지 말고, 필요한 경우에만 연구·교육용 참고 결과라고 짧게 말하세요.
""".strip()

        grounding_block = build_grounding_block(rag_results) if rag_results else ""
        user_content = (
            "아래 JSON 정보를 바탕으로 사용자의 질문에 답하세요.\n\n"
            + json.dumps(payload, ensure_ascii=False, indent=2)
        )
        if grounding_block:
            user_content += "\n\n# 참고 진료지침(이 내용을 우선 근거로 사용)\n" + grounding_block

        client = OpenAI()
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.2,
            max_tokens=700,
        )
        return resp.choices[0].message.content.strip(), rag_results

    except Exception as e:
        return (
            fallback_answer
            + f"\n\n※ LLM 답변 생성에 실패하여 규칙 기반 답변으로 표시했습니다. 오류: {e}",
            rag_results,
        )


# =============================================================================
# 화면 구성
# =============================================================================


# =============================================================================
# 화면 구성: 임상 workflow 중심
# =============================================================================

st.title("🫀 AI 기반 관상동맥질환 위험도 예측 임상 의사결정 지원 시스템")
st.caption(
    "환자 정보 수집 → 임상정보 구조화 → CAD 위험도 예측 → 위험요인 해석 → SOAP/EMR 기록 보조"
)

with st.sidebar:
    st.header("API 설정")
    st.caption("OpenAI API Key는 음성 전사와 SOAP 기록 초안 생성을 위해 사용됩니다.")

    configured_api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if configured_api_key:
        st.success("OpenAI API 연결 준비 완료")
        st.caption("저장된 API Key가 자동으로 적용되었습니다.")
        with st.expander("이번 실행에서 다른 API Key 사용", expanded=False):
            sidebar_api_key = st.text_input(
                "OpenAI API Key",
                value="",
                type="password",
                placeholder="sk-...",
                help="입력한 키는 현재 실행 세션에서만 사용되며 코드에는 저장되지 않습니다.",
            )
            if sidebar_api_key:
                os.environ["OPENAI_API_KEY"] = sidebar_api_key.strip()
                st.success("이번 실행에 사용할 API Key를 변경했습니다.")
    else:
        sidebar_api_key = st.text_input(
            "OpenAI API Key",
            value="",
            type="password",
            placeholder="sk-...",
            help="입력한 키는 현재 실행 세션에서만 사용되며 코드에는 저장되지 않습니다.",
        )
        if sidebar_api_key:
            os.environ["OPENAI_API_KEY"] = sidebar_api_key.strip()
            st.success("API Key 입력 완료")
        else:
            st.info("API Key를 입력하면 Whisper 음성 전사와 LLM 기반 SOAP 기록 초안 생성 기능을 사용할 수 있습니다.")

    st.divider()
    st.header("Workflow")
    st.markdown(
        """
        1. 환자 정보 수집  
        2. 임상정보 구조화  
        3. CAD 위험도 예측  
        4. 결과 해석 AI  
        5. SOAP/EMR 기록 보조  
        6. 예측 결과 다운로드
        """
    )

if "records" not in st.session_state:
    st.session_state.records = []

if "cad_model_service" not in st.session_state:
    st.session_state.cad_model_service = None
    st.session_state.model_error = None

if "emr_text_for_nlp" not in st.session_state:
    st.session_state["emr_text_for_nlp"] = ""

if "audio_text_for_nlp" not in st.session_state:
    st.session_state["audio_text_for_nlp"] = ""

if "clinical_transcript" not in st.session_state:
    st.session_state["clinical_transcript"] = ""

if "final_input_values" not in st.session_state:
    st.session_state["final_input_values"] = {}

if "final_input_confirmed" not in st.session_state:
    st.session_state["final_input_confirmed"] = False

# 입력 방식(radio)을 전환하면 직접 입력 위젯이 렌더링되지 않는다.
# Streamlit은 그리지 않은 위젯의 session_state 키를 자동으로 삭제하므로,
# 매 실행마다 input_ 접두사 키를 자기 자신에게 다시 할당해 직접 입력값을 유지한다.
# (버튼·파일업로더 등은 session_state로 설정할 수 없어 input_ 접두사로 한정한다.)
for _persist_key in list(st.session_state.keys()):
    if isinstance(_persist_key, str) and _persist_key.startswith("input_"):
        try:
            st.session_state[_persist_key] = st.session_state[_persist_key]
        except Exception:
            pass

# 모델은 기본 경로에서 자동 로드한다.
# 사용자가 화면에서 모델 경로를 직접 입력하지 않아도 되도록 구성했다.
if st.session_state.cad_model_service is None:
    try:
        default_model_path = find_default_model_path()
        if default_model_path and os.path.exists(default_model_path):
            st.session_state.cad_model_service = load_cad_model_service(default_model_path)
            st.session_state.model_path = default_model_path
            st.session_state.model_error = None
        else:
            st.session_state.model_error = (
                "최종 모델 파일을 찾을 수 없습니다. app.py와 같은 폴더에 "
                "cad_cdss_final_outputs 폴더를 두고 그 안의 "
                "cad_cdss_gradient_boosting_final.joblib 파일을 확인해 주세요."
            )
    except Exception as e:
        st.session_state.model_error = str(e)

if st.session_state.model_error:
    st.error(st.session_state.model_error)
    st.stop()

if st.session_state.cad_model_service is None:
    st.warning("CAD 예측 모델을 자동 로드하지 못했습니다. 모델 파일 위치를 확인해 주세요.")
    st.stop()

model_service = st.session_state.cad_model_service
model = model_service.model
threshold = model_service.threshold
feature_cols = list(model_service.feature_cols)
model_name = model_service.model_name

st.sidebar.divider()
st.sidebar.header("모델 상태")
st.sidebar.success("Gradient Boosting 기반 CAD 고위험군 분류 모델 연결 완료")
model_estimator = model.named_steps.get("model", model) if hasattr(model, "named_steps") else model
st.sidebar.caption(f"모델: {model_estimator.__class__.__name__}")
st.sidebar.caption(f"운영 임계값 {threshold:.2f} · 전체 입력 변수 {len(feature_cols)}개")
render_model_performance_sidebar()
# 탭 순서 자체를 임상 workflow에 맞춤
# 1번 탭에서 입력 방식을 먼저 선택하게 구성: 직접 입력 / EMR 텍스트 / 음성 입력
def next_step_hint(text: str):
    """탭 하단에 '다음 단계' 안내를 표시한다. (처음 사용하는 사람을 위한 흐름 안내)"""
    st.divider()
    st.info(f"➡️ {text}")


collection_tab, structure_tab, prediction_tab, interpretation_tab, soap_tab, data_tab = st.tabs([
    "1. 환자 정보 수집",
    "2. 임상정보 구조화",
    "3. CAD 위험도 예측",
    "4. 결과 해석 AI",
    "5. SOAP/EMR 기록 보조",
    "6. 예측 결과 다운로드",
])

values = {}

# =============================================================================
# 1. 환자 정보 수집: 직접 입력 / EMR 텍스트 / 음성 입력을 모두 받을 수 있게 구성
# =============================================================================
with collection_tab:
    st.subheader("1. 환자 정보 수집")
    st.markdown(
        """
        환자 정보를 수집합니다. 
        **직접 입력**, **EMR 자유 텍스트**, **음성 녹음/업로드**를 각각 입력할 수 있고,
        다음 단계에서 세 입력값을 비교한 뒤 **최종 모델 입력값 하나**로 병합합니다.
        """
    )

    mode_col1, mode_col2, mode_col3 = st.columns(3)
    with mode_col1:
        st.info("**직접 입력**\n\n나이, 성별, 혈압, 검사수치, 증상 여부를 의료진이 직접 입력합니다.")
    with mode_col2:
        st.info("**EMR 텍스트 입력**\n\n간호기록·진료 내용·검사 소견에서 CAD 관련 정보를 추출합니다.")
    with mode_col3:
        st.info("**음성 입력**\n\n진료 관련 음성 입력을 전사한 뒤 CAD 관련 정보를 추출합니다.")

    st.divider()

    # st.tabs는 text_area에서 Ctrl+Enter로 재실행될 때 첫 탭으로 돌아가는 경우가 있어,
    # 선택값이 session_state에 남는 radio 방식으로 입력 화면을 전환한다.
    input_mode = st.radio(
        "입력 방식 선택",
        ["직접 입력", "EMR 텍스트 입력", "음성 입력"],
        horizontal=True,
        key="collection_input_mode",
        label_visibility="collapsed",
    )

    # -------------------------------------------------------------------------
    # 1-A. 직접 입력
    # -------------------------------------------------------------------------
    if input_mode == "직접 입력":
        st.markdown("### 직접 입력")
        st.caption(
            "최종 모델이 사용하는 53개 변수를 실제 임상정보 확보 순서에 따라 입력합니다. "
            "확인되지 않은 값은 임의로 '없음'을 선택하지 말고 미선택 상태로 두세요."
        )

        st.info(
            "입력 단계: **1. 문진·과거력·활력징후·신체검사 → "
            "2. 혈액검사 → 3. 심전도 → 4. ECHO → 5. 결과 확인**"
        )

        # 문진·과거력·활력징후·신체검사 ------------------------------------
        with st.expander("1. 문진·과거력·활력징후·신체검사 (28개)", expanded=True):
            st.markdown("#### 기본정보")
            c1, c2, c3, c4 = st.columns(4)

            with c1:
                if "Sex" in feature_cols:
                    values["Sex"] = feature_input_widget("Sex")

            with c2:
                if "Age" in feature_cols:
                    values["Age"] = feature_input_widget("Age")

            with c3:
                height_cm = st.number_input(
                    "키(cm)",
                    min_value=0.0,
                    max_value=230.0,
                    value=0.0,
                    step=0.1,
                    key="input_height_cm_main",
                )

            with c4:
                weight_kg = st.number_input(
                    "몸무게(kg)",
                    min_value=0.0,
                    max_value=200.0,
                    value=0.0,
                    step=0.1,
                    key="input_weight_kg_main",
                )

            if "BMI" in feature_cols:
                if height_cm > 0 and weight_kg > 0:
                    height_m = height_cm / 100
                    bmi = round(weight_kg / (height_m ** 2), 1)
                    bmi_text = f"{bmi} kg/m²"
                else:
                    bmi = 0.0
                    bmi_text = "미입력"

                values["BMI"] = bmi
                st.caption(f"자동 계산 BMI: **{bmi_text}**")

            st.markdown("#### 과거력 및 위험인자")
            render_feature_grid(HISTORY_RISK_FEATURES, values, feature_cols)

            st.markdown("#### 증상")
            render_feature_grid(SYMPTOM_FEATURES, values, feature_cols)

            st.markdown("#### 활력징후 및 신체검사")
            if "BP" in feature_cols:
                values["BP"] = feature_input_widget("BP")
            if "PR" in feature_cols:
                values["PR"] = st.session_state.get(
                    "calculated_pr", DEFAULT_VALUES.get("PR", 0)
                )
            render_feature_grid(PHYSICAL_EXAM_FEATURES, values, feature_cols)

        # 혈액검사 -----------------------------------------------------------
        with st.expander("2. 혈액검사 (14개)", expanded=False):
            st.caption("현재 환자에게서 확보된 실제 검사값을 입력하세요.")
            render_feature_grid(LAB_FEATURES, values, feature_cols)

        # 심전도 -------------------------------------------------------------
        with st.expander("3. 심전도 (8개)", expanded=False):
            st.caption("12유도 ECG에서 확인된 소견을 입력하세요.")
            render_feature_grid(ECG_FEATURES, values, feature_cols)

        # 심초음파(ECHO) -----------------------------------------------------
        with st.expander("4. 심초음파(ECHO) (3개, 결과 확보 시 입력)", expanded=False):
            st.caption(
                "ECHO를 시행하지 않았거나 결과가 아직 확인되지 않았다면 미입력 상태로 두세요. "
                "결과 화면에서 누락 항목과 대체 여부를 별도로 안내합니다."
            )
            render_feature_grid(ECHO_FEATURES, values, feature_cols)

        leftover = [f for f in feature_cols if f not in values]
        if leftover:
            with st.expander("기타 모델 입력 변수", expanded=False):
                st.warning(
                    "저장된 모델에 현재 화면 분류표에 없는 변수가 있습니다. "
                    "모델 파일 또는 변수군 정의가 변경됐는지 확인하세요."
                )
                render_feature_grid(leftover, values, feature_cols)

        st.caption(
            "입력을 마친 뒤 상단의 **2. 임상정보 구조화**에서 EMR·음성 추출값과 비교하거나, "
            "**3. CAD 위험도 예측**에서 현재 입력값 기준 결과를 확인할 수 있습니다."
        )

    # -------------------------------------------------------------------------
    # 1-B. EMR 자유 텍스트
    # -------------------------------------------------------------------------
    if input_mode == "EMR 텍스트 입력":
        st.markdown("### EMR 텍스트 입력")
        st.caption(
            "간호기록, 진료 내용, 검사 소견 등 CAD 관련 임상 정보를 입력하세요."
        )

        with st.expander("ℹ️ 이 텍스트는 어떻게 분석되나요?", expanded=False):
            st.markdown(
                "두 단계로 변수를 추출합니다.\n\n"
                "1. **정규식·사전**: 나이·혈압 등 수치와 '고혈압', '흉통' 같은 표준 표현을 찾습니다. "
                "(주변에 '없음/부인'이 있으면 제외)\n"
                "2. **의미 유사도 매칭**: 사전에 없는 표현도 글자 단위 유사도로 연결합니다. "
                "예) \"가슴이 쥐어짜듯 아프다\"→전형적 흉통, \"숨이 가쁘다\"→호흡곤란.\n\n"
                "아래 표의 **추출 방식** 열에서 어느 단계로 잡혔는지 볼 수 있습니다."
            )

        st.text_area(
            "임상 텍스트 입력",
            key="emr_text_for_nlp",
            height=240,
        )

        with st.expander("입력 예시", expanded=False):
            st.code(
                "60세 남성. 계단 오를 때 가슴이 쥐어짜듯 아프다고 함.\n"
                "숨이 가쁘다고 하며 혈압이 높다고 함. ECG상 T파 역전 의심. EF 45%, RWMA 소견 있음.",
                language="text",
            )
            st.caption(
                "'가슴이 쥐어짜듯 아프다', '숨이 가쁘다'는 사전에 없는 표현이지만 "
                "의미 유사도 매칭으로 각각 전형적 흉통·호흡곤란으로 잡힙니다."
            )

        emr_input_text = st.session_state.get("emr_text_for_nlp", "")
        preview_emr_extracted = extract_clinical_features_from_text(emr_input_text)
        if preview_emr_extracted:
            # 정규식/사전만으로 잡힌 결과와 비교해 '추출 방식'을 표시한다.
            exact_only = extract_clinical_features_from_text(emr_input_text, use_semantic=False)
            method_rows = []
            for key, value in preview_emr_extracted.items():
                method = "정규식·사전" if key in exact_only else "유사도 매칭"
                method_rows.append({
                    "추출 변수": key,
                    "한글명": VARIABLE_NAME_KR.get(key, key),
                    "추출값": value,
                    "추출 방식": method,
                    "모델 반영 방식": "이진/범주" if key in BINARY_FEATURES or key in ORDINAL_FEATURES else "수치",
                })
            st.markdown("#### EMR 텍스트에서 미리 추출된 정보")
            st.dataframe(pd.DataFrame(method_rows), use_container_width=True, hide_index=True)
            if any(r["추출 방식"] == "유사도 매칭" for r in method_rows):
                st.caption("🔎 '유사도 매칭'으로 표시된 변수는 사전에 없는 자연어 표현을 의미 유사도로 잡아낸 항목입니다.")
        else:
            st.info("현재 EMR 텍스트에서 추출된 CAD 관련 정보가 없습니다.")

    # -------------------------------------------------------------------------
    # 1-C. 음성 녹음 / 업로드
    # -------------------------------------------------------------------------
    if input_mode == "음성 입력":
        st.markdown("### 음성 입력")
        st.caption(
            "진료 관련 음성 입력을 Whisper 기반 전사 기능으로 텍스트로 변환합니다."
        )

        audio_col1, audio_col2 = st.columns(2)

        recorded_audio = None
        with audio_col1:
            if hasattr(st, "audio_input"):
                recorded_audio = st.audio_input("마이크로 진료 관련 음성 녹음", key="clinical_audio_input")
            else:
                st.info("현재 Streamlit 버전에는 st.audio_input이 없습니다. Streamlit 업데이트 또는 오디오 파일 업로드를 사용하세요.")

        with audio_col2:
            uploaded_audio = st.file_uploader(
                "오디오 파일 업로드",
                type=["wav", "mp3", "m4a", "webm", "ogg", "mp4"],
                key="clinical_audio_uploader",
            )

        chosen_audio = recorded_audio or uploaded_audio
        if chosen_audio is not None:
            st.audio(chosen_audio)
            if st.button("Whisper로 음성 전사", type="primary", key="transcribe_button"):
                try:
                    with st.spinner("음성을 텍스트로 전사하는 중..."):
                        audio_bytes = chosen_audio.getvalue()
                        file_name = getattr(chosen_audio, "name", "clinical_audio.wav")
                        transcript = transcribe_audio_with_openai(audio_bytes, file_name=file_name)
                        st.session_state["clinical_transcript"] = transcript
                        st.session_state["audio_text_for_nlp"] = transcript
                    st.success("전사 완료. 음성 입력값으로 저장했습니다.")
                except Exception as e:
                    st.error(f"전사 실패: {e}")

        st.markdown("#### 음성 전사 텍스트")
        st.text_area(
            "전사 결과 확인/수정",
            key="audio_text_for_nlp",
            height=180,
        )

        preview_audio_extracted = extract_clinical_features_from_text(st.session_state.get("audio_text_for_nlp", ""))
        if preview_audio_extracted:
            st.markdown("#### 음성 전사에서 미리 추출된 정보")
            st.dataframe(make_extracted_features_dataframe(preview_audio_extracted), use_container_width=True, hide_index=True)
        else:
            st.info("음성 전사 텍스트에서 추출된 CAD 관련 정보가 없습니다.")

    next_step_hint("입력을 마쳤다면 상단 **2. 임상정보 구조화** 탭으로 이동해 직접·EMR·음성 값을 비교하고 최종값을 확정하세요.")

# radio 전환/텍스트 입력 재실행 이후에도 직접 입력값이 사라지지 않도록 복원한다.
values = {**collect_manual_values_from_session(feature_cols), **values}

# =============================================================================
# 2. 임상정보 구조화: 직접 입력 + EMR 추출 + 음성 추출을 비교하고 최종값 확정
# =============================================================================
with structure_tab:
    st.subheader("2. 임상정보 구조화")
    st.write(
        "직접 입력값, EMR 텍스트 추출값, 음성 전사 추출값을 한 표에서 비교합니다. "
        "같은 변수가 서로 다르면 의료진이 확인한 뒤 최종 적용값을 확정합니다."
    )

    emr_text = st.session_state.get("emr_text_for_nlp", "")
    audio_text = st.session_state.get("audio_text_for_nlp", "")
    emr_extracted = {k: v for k, v in extract_clinical_features_from_text(emr_text).items() if k in feature_cols}
    audio_extracted = {k: v for k, v in extract_clinical_features_from_text(audio_text).items() if k in feature_cols}

    st.markdown("### 입력 원문 확인")
    t1, t2 = st.columns(2)
    with t1:
        st.text_area("EMR 자유 텍스트", value=emr_text, height=150, disabled=True)
    with t2:
        st.text_area("음성 전사 텍스트", value=audio_text, height=150, disabled=True)

    # 53개 모델 변수를 직접 입력 화면과 같은 임상 순서로 모두 표시한다.
    # 화면 순서와 실제 모델 입력 순서는 다를 수 있으며, 예측 직전에 서비스가
    # 저장된 feature_cols 순서로 다시 정렬한다.
    all_show_features = [
        feature for feature in CLINICAL_WORKFLOW_ORDER if feature in feature_cols
    ]
    all_show_features.extend(
        feature for feature in feature_cols if feature not in all_show_features
    )

    # 추천값 우선순위: 의료진이 직접 확인해 입력한 값 > EMR 추출 > 음성 추출.
    # NLP 추출값은 자동 매칭 결과이므로, 사람이 명시적으로 입력한 값이 있으면 그 값을 우선한다.
    # 단 '미입력' 상태(연속형 0, 미선택 None)는 값이 없는 것으로 보고 추출값으로 채운다.
    def _is_entered(feature: str, value) -> bool:
        if value is None or value == "":
            return False
        try:
            if pd.isna(value):
                return False
        except (TypeError, ValueError):
            pass
        if feature in CONTINUOUS_FEATURES:
            try:
                return float(value) != 0
            except (TypeError, ValueError):
                return False
        return True  # 이진·서열형의 0은 의료진이 고른 '없음/정상'이므로 유효

    recommended_values = {}
    recommended_source = {}
    for k, v in audio_extracted.items():
        recommended_values[k] = v
        recommended_source[k] = "음성 추출"
    for k, v in emr_extracted.items():
        recommended_values[k] = v
        recommended_source[k] = "EMR 추출"
    for k, v in values.items():
        if _is_entered(k, v):
            recommended_values[k] = v
            recommended_source[k] = "직접 입력"
        elif k not in recommended_values:
            recommended_values[k] = v
            recommended_source[k] = "직접 입력"

    rows = []
    for f in all_show_features:
        direct_v = values.get(f, DEFAULT_VALUES.get(f, 0))
        emr_v = emr_extracted.get(f, "")
        audio_v = audio_extracted.get(f, "")
        rec_v = recommended_values.get(f, DEFAULT_VALUES.get(f, 0))
        candidates = [str(x) for x in [direct_v, emr_v, audio_v] if x != ""]
        conflict = "충돌" if len(set(candidates)) > 1 else ""
        rows.append({
            "변수": f,
            "한글명": VARIABLE_NAME_KR.get(f, f),
            "직접 입력": direct_v,
            "EMR 추출": emr_v,
            "음성 추출": audio_v,
            "최종 적용값": rec_v,
        })

    st.markdown("### 53개 변수 직접 입력 · EMR · 음성 비교표")
    st.info(
        "표의 **최종 적용값** 컬럼이 실제 CAD 예측 모델에 들어갈 값입니다. "
        "값이 틀리면 이 칸에서 직접 수정한 뒤, 아래 **이 값으로 최종 모델 입력 확정** 버튼을 누르세요."
    )

    if rows:
        compare_df = pd.DataFrame(rows)

        # Streamlit data_editor는 컬럼의 실제 dtype과 column_config 타입이 맞아야 합니다.
        # 최종 적용값은 사용자가 직접 수정할 수 있게 TextColumn으로 쓰기 때문에,
        # 숫자/문자 혼합 오류를 막기 위해 표시용 컬럼을 문자열로 통일합니다.
        display_cols = ["직접 입력", "EMR 추출", "음성 추출", "최종 적용값"]
        for col in display_cols:
            if col in compare_df.columns:
                compare_df[col] = compare_df[col].apply(lambda x: "" if pd.isna(x) else str(x))

        edited_df = st.data_editor(
            compare_df,
            use_container_width=True,
            hide_index=True,
            disabled=["변수", "한글명", "직접 입력", "EMR 추출", "음성 추출"],
            column_config={
                "최종 적용값": st.column_config.TextColumn(
                    "최종 적용값",
                    help="실제 CAD 예측 모델에 입력될 값입니다. 필요하면 직접 수정하세요.",
                    width="medium",
                ),
                "변수": st.column_config.TextColumn("변수", width="small"),
                "한글명": st.column_config.TextColumn("한글명", width="medium"),
                "직접 입력": st.column_config.TextColumn("직접 입력", width="small"),
                "EMR 추출": st.column_config.TextColumn("EMR 추출", width="small"),
                "음성 추출": st.column_config.TextColumn("음성 추출", width="small"),
            },
            key="final_merge_editor",
        )

        c_reset, c_apply = st.columns([1, 1])
        with c_reset:
            if st.button("최종 확정값 초기화", key="reset_final_values"):
                st.session_state["final_input_values"] = {}
                st.session_state["final_input_confirmed"] = False
                st.info("최종 확정값을 초기화했습니다. 현재 직접 입력값 기준으로 돌아갑니다.")
        with c_apply:
            if st.button("이 값으로 최종 모델 입력 확정", type="primary", key="confirm_merged_values"):
                final_confirmed = dict(values)
                for _, row in edited_df.iterrows():
                    feature = row["변수"]
                    raw_value = row["최종 적용값"]
                    try:
                        if feature in BINARY_FEATURES or feature in ORDINAL_FEATURES or feature in {"Age"}:
                            final_confirmed[feature] = int(float(raw_value))
                        else:
                            final_confirmed[feature] = float(raw_value)
                    except Exception:
                        # 변환 실패 시 추천값 또는 직접 입력값 유지
                        final_confirmed[feature] = recommended_values.get(feature, values.get(feature, DEFAULT_VALUES.get(feature, 0)))
                st.session_state["final_input_values"] = {k: v for k, v in final_confirmed.items() if k in feature_cols}
                st.session_state["final_input_confirmed"] = True
                st.success("최종 모델 입력값을 확정했습니다. 상단의 3. CAD 위험도 예측 탭으로 이동해 결과를 확인하세요.")
    else:
        st.info("표시할 입력 변수가 없습니다. 1번 탭에서 직접 입력하거나 EMR/음성을 입력하세요.")

    if st.session_state.get("final_input_confirmed") and st.session_state.get("final_input_values"):
        st.markdown("### 현재 확정된 최종 입력값")
        confirmed_df = make_input_dataframe(st.session_state["final_input_values"], feature_cols)
        st.dataframe(confirmed_df, use_container_width=True)
    else:
        st.caption("아직 최종값이 확정되지 않았습니다. 3번 예측은 현재 직접 입력값 기준으로 계산됩니다.")

    next_step_hint("최종값을 확정했다면 상단 **3. CAD 위험도 예측** 탭으로 이동해 결과를 확인하세요.")

# 최종 입력값 결정: 확정값이 있으면 확정값 사용, 없으면 직접 입력값 사용
if st.session_state.get("final_input_confirmed") and st.session_state.get("final_input_values"):
    final_values = dict(values)
    final_values.update({k: v for k, v in st.session_state["final_input_values"].items() if k in feature_cols})
    final_source_label = "직접 입력 + EMR/음성 병합 확정값"
else:
    final_values = dict(values)
    final_source_label = "직접 입력값"

service_values = prepare_values_for_service(final_values, feature_cols)
# 기존 규칙 기반 요약 함수는 0을 '표시하지 않음'으로 사용하므로, 화면용 설명에만
# 결측값을 0으로 바꾼 사본을 전달한다. 실제 모델 입력은 service_values를 사용한다.
legacy_values = {
    col: (0 if service_values.get(col) is None or pd.isna(service_values.get(col)) else service_values.get(col))
    for col in feature_cols
}
input_df = make_input_dataframe(service_values, feature_cols)
model_input_df, missing_report_obj = model_service.prepare_input(service_values)
missing_report = missing_report_obj.to_dict()
input_ready = has_meaningful_input(service_values, feature_cols)
prediction_result = None

if input_ready:
    try:
        prediction_result = model_service.predict(service_values).to_dict()
        probability = float(prediction_result["cad_score"])
    except Exception as e:
        st.error(f"예측 실패: {e}")
        st.stop()

    risk_group, risk_color = get_risk_level(probability, threshold)
    reasons = make_reason_list(legacy_values)
    emr_summary = make_emr_summary(legacy_values, probability, threshold, risk_group)
else:
    probability = None
    risk_group = "입력 대기"
    risk_color = "#94a3b8"
    reasons = []
    emr_summary = "환자 정보를 입력하면 요약이 생성됩니다."

# =============================================================================
# 3. CAD 위험도 예측
# =============================================================================
with prediction_tab:
    st.subheader("3. CAD 위험도 예측")
    st.write(f"현재 예측 기준: **{final_source_label}**")

    if not input_ready:
        st.info("환자 정보를 입력하면 CAD 위험도 예측 결과가 표시됩니다.")
        st.markdown("### 최종 모델 입력 데이터")
        st.dataframe(input_df, use_container_width=True)
        st.warning("주의: 이 결과는 연구/교육용 CDSS 프로토타입의 참고 결과이며 실제 진단을 대체하지 않습니다.")
    else:
        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            st.metric("CAD 분류 점수", f"{probability:.1%}")
        with c2:
            st.metric("입력 완성도", f"{prediction_result['completeness']:.1%}")
            st.caption(f"{prediction_result['entered_features']} / {prediction_result['total_features']}개 입력")
        with c3:
            render_risk_badge(probability, threshold)

        st.progress(min(max(probability, 0.0), 1.0))

        if prediction_result["missing_count"]:
            st.warning(
                f"실제 환자값이 입력되지 않은 항목이 {prediction_result['missing_count']}개 있습니다. "
                "해당 값은 저장된 모델 Pipeline에서 학습 데이터 중앙값으로 대체되었습니다."
            )
            with st.expander("미입력 항목 확인", expanded=False):
                for group_name, missing_names in prediction_result["missing_by_group_kr"].items():
                    if missing_names:
                        st.markdown(f"**{group_name}**: {', '.join(missing_names)}")

        if prediction_result["completeness"] < 0.80:
            st.error(
                "입력 완성도가 80% 미만입니다. 결과의 불확실성이 커질 수 있습니다. "
                "미입력은 실제 '없음'과 다릅니다. 확인 가능한 실제 문진·검사값을 추가하면 "
                "현재 환자 정보에 근거한 해석을 더 충실하게 할 수 있습니다."
            )

        if st.session_state.get("final_input_confirmed"):
            st.success("직접 입력·EMR·음성 입력값을 비교한 최종 확정값이 예측에 반영되어 있습니다.")
        else:
            st.info("현재 예측은 직접 입력값 기준입니다. 2번 탭에서 EMR/음성 추출값을 비교하고 최종값을 확정할 수 있습니다.")

        st.markdown("### 최종 모델 입력 데이터")
        st.dataframe(input_df, use_container_width=True)

        st.markdown("### 예측 근거 해석")
        st.caption("현재 입력값에서 CAD 위험도 판단에 참고할 수 있는 항목을 의료진이 보기 쉽게 정리합니다.")

        if reasons:
            st.markdown("**현재 입력값 기준 주요 참고 요인**")
            reason_cols = st.columns(3)
            for idx, reason in enumerate(reasons[:6]):
                with reason_cols[idx % 3]:
                    st.info(reason)
        else:
            st.info("현재 입력값 기준으로 별도로 강조되는 위험요인은 제한적입니다.")

        st.caption(
            "미입력 항목은 모델 Pipeline이 학습 데이터 중앙값으로 대체합니다. "
            "대체값은 실제 환자 검사값이 아니며, 위 입력 완성도와 함께 해석해야 합니다."
        )
        xai_df, shap_error = explain_prediction(model, model_input_df, feature_cols)

        if not xai_df.empty:
            explanation_method = str(xai_df.iloc[0].get("설명 방식", ""))
            if explanation_method == "SHAP":
                st.markdown("**현재 환자 SHAP 영향도 Top 5**")
                st.caption("양수는 CAD 고위험 점수를 높이는 방향, 음수는 낮추는 방향의 모델 기여도입니다.")
            else:
                st.markdown("**전체 모델 변수 중요도 Top 5**")
                st.warning(
                    "개별 환자 SHAP 값을 계산하지 못해 Gradient Boosting의 전체 변수 중요도를 표시합니다. "
                    f"세부 원인: {shap_error}"
                )
            display_cols = ["한글명", "변수", "입력값", "영향 방향", "영향도"]
            top_xai = xai_df.head(5)[[c for c in display_cols if c in xai_df.columns]].copy()
            if "영향도" in top_xai.columns:
                top_xai["영향도"] = pd.to_numeric(top_xai["영향도"], errors="coerce").round(4)
            st.dataframe(top_xai, use_container_width=True, hide_index=True)

            render_xai_bar_chart(xai_df, top_n=5)
        else:
            st.info(
                "현재 환자의 모델 판단 영향도를 계산하지 못했습니다. "
                f"세부 원인: {shap_error or '확인되지 않은 오류'}"
            )

        st.warning("주의: 이 결과는 연구/교육용 CDSS 프로토타입의 참고 결과이며 실제 진단을 대체하지 않습니다.")

    next_step_hint("결과를 확인했다면 상단 **4. 결과 해석 AI** 탭에서 질문하거나, **5. SOAP/EMR 기록 보조** 탭에서 기록 초안을 생성하세요.")

# =============================================================================
# 4. 결과 해석
# =============================================================================
with interpretation_tab:
    st.subheader("4. 결과 해석 AI")

    llm_available = bool(os.getenv("OPENAI_API_KEY"))
    use_llm_chat = llm_available and input_ready
    chat_model = "gpt-4o-mini"

    # 진료지침 RAG 검색기 (키 유무에 따라 OpenAI 임베딩 / TF-IDF 폴백)
    retriever = get_guideline_retriever(llm_available)
    if retriever is not None:
        st.caption(
            f"📚 진료지침 RAG 활성화 — 백엔드: {retriever.backend} · "
            f"문서 {retriever.num_docs}개 / 청크 {retriever.num_chunks}개"
        )
    else:
        st.caption("📚 진료지침 RAG 비활성화 (guidelines 폴더/모듈 없음) — 일반 답변으로 동작합니다.")

    with st.expander("📖 진료지침 RAG 참고 출처", expanded=False):
        st.markdown(
            "RAG가 검색·인용하는 진료지침 문서의 근거 출처입니다. "
            "(정확한 연도·권·페이지·DOI는 각 원문에서 확인하세요.)\n\n"
            "- **01 위험요인**: 2019 ESC Chronic Coronary Syndromes 지침 / 2013 ACC/AHA Cardiovascular Risk Assessment 지침 / ADA Standards of Care in Diabetes / Z-Alizadeh Sani 데이터셋 논문(Alizadehsani 등)\n"
            "- **02 흉통**: 2021 AHA/ACC Chest Pain 평가 지침 / 2019 ESC CCS 지침 / Diamond-Forrester·CAD Consortium 사전확률 모델\n"
            "- **03 심전도**: 2018 Fourth Universal Definition of Myocardial Infarction(ESC/ACC/AHA/WHF) / 2021 AHA/ACC Chest Pain 지침\n"
            "- **04 심초음파**: ASE/EACVI Cardiac Chamber Quantification 권고 / 2021 AHA/ACC Chest Pain 지침 / ACC·ESC 심부전 지침\n"
            "- **05 지질·당뇨·혈압**: 2018 AHA/ACC Cholesterol 지침(또는 2019 ESC/EAS Dyslipidaemia) / ADA Standards of Care / 2017 ACC/AHA(또는 2018 ESC/ESH) 고혈압 지침\n"
            "- **06 추가검사·의뢰**: 2021 AHA/ACC Chest Pain 지침 / 2019 ESC CCS 지침 / 고민감도 트로포닌 기반 ACS 평가 지침"
        )

    if llm_available:
        st.success("OpenAI API Key가 입력되어 있어 LLM 기반 답변을 사용합니다.")
        with st.expander("고급 설정", expanded=False):
            chat_model = st.selectbox(
                "챗봇 모델",
                ["gpt-4o-mini", "gpt-4o"],
                index=0,
                key="cdss_chat_model",
            )
    else:
        st.info("OpenAI API Key가 없어서 규칙 기반 답변으로 동작합니다. API Key를 입력하면 LLM 기반 답변을 사용할 수 있습니다.")

    if "cdss_chat_messages" not in st.session_state:
        st.session_state["cdss_chat_messages"] = [
            {
                "role": "assistant",
                "content": (
                    "CAD 위험도 예측 결과를 임상적으로 해석해 드립니다. "
                    "환자 정보를 입력하면 CAD 모델 점수, 참고군, 주요 위험요인, 추가 확인사항을 짧게 정리해 드립니다."
                ),
            }
        ]

    st.markdown("#### 빠른 질문")
    sample_col1, sample_col2, sample_col3 = st.columns([1, 1, 1])
    sample_question = None

    with sample_col1:
        if st.button("위험요인 설명", key="sample_risk_factor", use_container_width=True):
            sample_question = "현재 입력값 기준으로 위험요인을 설명해 주세요."
    with sample_col2:
        if st.button("추가 확인사항", key="sample_additional_check", use_container_width=True):
            sample_question = "추가로 확인하면 좋은 항목은 무엇인가요?"
    with sample_col3:
        if st.button("기록용 요약", key="sample_summary_sentence", use_container_width=True):
            sample_question = "현재 예측 결과를 의료진 참고용으로 짧게 요약해 주세요."

    st.divider()

    for msg in st.session_state["cdss_chat_messages"]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_question = st.chat_input("질문을 입력하세요. 예: 이 환자에서 추가로 확인할 항목은?", key="cdss_chat_input")

    if sample_question:
        user_question = sample_question

    if user_question:
        st.session_state["cdss_chat_messages"].append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.write(user_question)

        with st.chat_message("assistant"):
            with st.spinner("답변 생성 중..."):
                rag_sources: list[dict] = []
                if not input_ready:
                    answer = (
                        "아직 환자 정보가 입력되지 않아 현재 환자에 대한 해석은 제공할 수 없습니다. "
                        "1번 탭에서 환자 정보를 입력하거나 EMR/음성 입력을 반영한 뒤 다시 질문해 주세요."
                    )
                else:
                    answer, rag_sources = generate_cdss_chat_answer(
                        user_question=user_question,
                        values=legacy_values,
                        probability=probability,
                        threshold=threshold,
                        risk_group=risk_group,
                        reasons=reasons,
                        emr_summary=emr_summary,
                        use_llm=use_llm_chat,
                        model_name=chat_model,
                        retriever=retriever,
                    )
                st.write(answer)
                if rag_sources:
                    with st.expander(f"📚 참고한 진료지침 근거 {len(rag_sources)}건", expanded=False):
                        for s in rag_sources:
                            st.markdown(f"**{s['source']}** · 유사도 {s['score']}")
                            st.caption(s["text"])

        st.session_state["cdss_chat_messages"].append({"role": "assistant", "content": answer})

    reset_col, _ = st.columns([1, 5])
    with reset_col:
        if st.button("대화 초기화", key="reset_cdss_chat", use_container_width=True):
            st.session_state["cdss_chat_messages"] = [
                {
                    "role": "assistant",
                    "content": "대화를 초기화했습니다. 현재 CAD 예측 결과에 대해 다시 질문해 주세요.",
                }
            ]
            st.rerun()

    next_step_hint("상단 **5. SOAP/EMR 기록 보조** 탭에서 요약·SOAP 초안을 생성해 EMR에 자동 반영할 수 있습니다.")

# =============================================================================
# 5. SOAP/EMR 기록 보조
# =============================================================================
with soap_tab:
    st.subheader("5. SOAP/EMR 기록 보조")
    st.write("예측 결과와 임상 텍스트를 바탕으로 의료진 참고용 SOAP/EMR 요약을 생성합니다.")
    st.info(
        "초안이 생성되면 별도의 복사·붙여넣기 없이 **EMR 입력 필드에 자동으로 반영**됩니다. "
        "아래 'EMR 시스템에 자동 입력' 버튼을 누르면 실제 병원 EMR 연동 모듈로 전송되는 구조로 동작합니다."
    )

    st.markdown("### 입력 정보 요약")

    sum_gen_col, sum_info_col = st.columns([1, 2])
    with sum_gen_col:
        if st.button("🧠 RAG/LLM으로 요약 생성", key="generate_llm_summary_button"):
            if not input_ready:
                st.warning("환자 정보를 먼저 입력한 뒤 요약을 생성하세요.")
            else:
                try:
                    with st.spinner("진료지침 RAG + LLM 요약 생성 중..."):
                        summary_clinical_text = "\n".join([
                            "[EMR 텍스트]", st.session_state.get("emr_text_for_nlp", ""),
                            "[음성 전사]", st.session_state.get("audio_text_for_nlp", ""),
                        ])
                        summary_extracted = extract_clinical_features_from_text(summary_clinical_text)
                        summary_retriever = get_guideline_retriever(bool(os.getenv("OPENAI_API_KEY")))
                        summary_text, summary_sources = generate_llm_input_summary(
                            clinical_text=summary_clinical_text,
                            values=legacy_values,
                            probability=probability,
                            threshold=threshold,
                            risk_group=risk_group,
                            extracted=summary_extracted,
                            model_name=st.session_state.get("llm_soap_model", "gpt-4o-mini"),
                            retriever=summary_retriever,
                            reasons=reasons,
                        )
                        st.session_state["llm_input_summary"] = summary_text
                        st.session_state["llm_input_summary_sources"] = summary_sources
                    st.success("RAG/LLM 요약 생성 완료")
                except Exception as e:
                    st.error(f"RAG/LLM 요약 생성 실패: {e}")
    with sum_info_col:
        if st.session_state.get("llm_input_summary"):
            st.caption("현재 표시: 🧠 진료지침 RAG + LLM 생성 요약 (읽기 전용)")
        else:
            st.caption("현재 표시: 규칙 기반 자동 요약. 위 버튼을 누르면 진료지침 RAG + LLM 요약으로 대체됩니다. (읽기 전용)")

    display_summary = st.session_state.get("llm_input_summary") or emr_summary
    st.text_area(
        "요약 내용",
        value=display_summary,
        height=200,
        disabled=True,
        label_visibility="collapsed",
    )

    summary_sources = st.session_state.get("llm_input_summary_sources") or []
    if st.session_state.get("llm_input_summary"):
        if summary_sources:
            with st.expander(f"📚 요약에 참고한 진료지침 근거 {len(summary_sources)}건", expanded=False):
                for s in summary_sources:
                    st.markdown(f"**{s['source']}** · 유사도 {s['score']}")
                    st.caption(s["text"])
        if st.button("규칙 기반 요약으로 되돌리기", key="reset_llm_summary"):
            st.session_state.pop("llm_input_summary", None)
            st.session_state.pop("llm_input_summary_sources", None)
            st.rerun()

    st.markdown("### LLM 기반 SOAP 기록 초안 생성")
    llm_model = st.selectbox("LLM 모델", ["gpt-4o-mini", "gpt-4o"], index=0, key="llm_soap_model")

    if st.button("LLM으로 SOAP/EMR 기록 초안 생성", type="primary", key="generate_llm_soap_button"):
        if not input_ready:
            st.warning("환자 정보를 먼저 입력한 뒤 SOAP/EMR 기록 초안을 생성하세요.")
        else:
            try:
                with st.spinner("SOAP/EMR 기록 초안 생성 중..."):
                    combined_clinical_text = "\n".join([
                        "[EMR 텍스트]", st.session_state.get("emr_text_for_nlp", ""),
                        "[음성 전사]", st.session_state.get("audio_text_for_nlp", ""),
                    ])
                    current_extracted = extract_clinical_features_from_text(combined_clinical_text)
                    soap_retriever = get_guideline_retriever(bool(os.getenv("OPENAI_API_KEY")))
                    soap_note_text, soap_rag_sources = generate_llm_soap_note(
                        clinical_text=combined_clinical_text,
                        values=legacy_values,
                        probability=probability,
                        threshold=threshold,
                        risk_group=risk_group,
                        extracted=current_extracted,
                        model_name=llm_model,
                        retriever=soap_retriever,
                        reasons=reasons,
                    )
                    st.session_state["llm_soap_note"] = soap_note_text
                    st.session_state["llm_soap_sources"] = soap_rag_sources
                    # 생성 즉시 EMR 입력 필드(편집 가능)에 자동 반영 (복사/붙여넣기 불필요)
                    st.session_state["emr_auto_insert_preview"] = soap_note_text
                    st.session_state["emr_auto_insert_text"] = soap_note_text
                    # 새 초안이 생성되면 직전 전송 상태는 초기화
                    st.session_state["emr_insert_status"] = None
                st.success("SOAP/EMR 기록 초안 생성 완료 — EMR 입력 필드에 자동 반영되었습니다.")
            except Exception as e:
                st.error(f"LLM SOAP 기록 초안 생성 실패: {e}")

    if st.session_state.get("llm_soap_note"):
        st.markdown("### EMR 입력 필드 (자동 반영됨)")
        st.caption(
            "위에서 생성된 초안이 이 필드에 자동으로 입력되어 있습니다. "
            "필요하면 수정한 뒤 EMR 시스템으로 전송할 수 있습니다."
        )
        st.text_area(
            "EMR 자동 입력 보조 영역",
            height=360,
            key="emr_auto_insert_preview",
            label_visibility="collapsed",
            help="실제 병원 EMR API/연동 모듈이 있다면 이 영역의 내용이 자동 입력 대상으로 전달되는 구조입니다.",
        )

        insert_col1, insert_col2 = st.columns([1, 3])
        with insert_col1:
            if st.button("📤 EMR 시스템에 자동 입력", type="primary", key="push_to_emr_button"):
                st.session_state["emr_insert_status"] = {
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "chars": len(st.session_state.get("emr_auto_insert_preview", "")),
                }
        with insert_col2:
            status = st.session_state.get("emr_insert_status")
            if status:
                st.success(
                    f"✅ EMR 시스템 자동 입력 완료 · {status['time']} · {status['chars']}자 전송됨"
                )
            else:
                st.caption("버튼을 누르면 위 내용이 EMR 입력 모듈로 자동 전송됩니다.")

        soap_sources = st.session_state.get("llm_soap_sources") or []
        if soap_sources:
            with st.expander(f"📚 참고한 진료지침 근거 {len(soap_sources)}건", expanded=False):
                for s in soap_sources:
                    st.markdown(f"**{s['source']}** · 유사도 {s['score']}")
                    st.caption(s["text"])
    else:
        st.info("OPENAI_API_KEY 설정 후 버튼을 누르면 LLM 기반 SOAP/EMR 기록 초안이 생성되어 EMR 입력 필드에 자동 반영됩니다.")

    st.caption("임상 배포용이 아니라 연구/교육용 보조 기능입니다. 실제 진단·처방 결정은 의료진 판단이 필요합니다.")

    next_step_hint("필요하면 상단 **6. 예측 결과 다운로드** 탭에서 결과를 저장하고 CSV로 내려받으세요.")

# =============================================================================
# 6. 예측 결과 다운로드
# =============================================================================
with data_tab:
    st.subheader("6. 예측 결과 다운로드")
    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model_name": model_name,
        "threshold": threshold,
        "cad_model_score": probability,
        "risk_group": risk_group,
        "input_completeness": prediction_result.get("completeness") if prediction_result else None,
        "missing_count": prediction_result.get("missing_count") if prediction_result else len(feature_cols),
        "missing_features": ", ".join(prediction_result.get("missing_features", [])) if prediction_result else "",
        "merged_input_confirmed": bool(st.session_state.get("final_input_confirmed")),
        **{col: input_df.iloc[0][col] for col in feature_cols},
    }

    if st.button("현재 예측 결과를 세션에 저장"):
        st.session_state.records.append(record)
        st.success("저장 완료")

    if st.session_state.records:
        records_df = pd.DataFrame(st.session_state.records)
        st.dataframe(records_df, use_container_width=True)
        # Excel에서 한글이 깨지지 않도록 UTF-8 BOM(utf-8-sig) 바이트로 내려준다.
        csv_bytes = records_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="예측 결과 CSV 다운로드",
            data=csv_bytes,
            file_name="cad_cdss_prediction_results.csv",
            mime="text/csv; charset=utf-8-sig",
        )
    else:
        st.info("아직 저장된 예측 결과가 없습니다.")

st.divider()
st.caption("관상동맥질환 위험도 예측 CDSS 프로토타입입니다. 본 결과는 연구·교육용 참고자료이며, 실제 진단이나 치료 결정을 대체하지 않습니다.")
