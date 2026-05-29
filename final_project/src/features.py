import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

TARGET_MAP = {"Dropout": 0, "Enrolled": 1, "Graduate": 2}
TARGET_MAP_INV = {v: k for k, v in TARGET_MAP.items()}

# 모델에 사용할 주요 특성 (설명 가능한 변수만 선택)
KEY_FEATURES = [
    "Age at enrollment",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Tuition fees up to date",
    "Scholarship holder",
    "Debtor",
    "Gender",
    "Displaced",
    "Educational special needs",
    "International",
    "Unemployment rate",
    "Inflation rate",
    "GDP",
    "Application mode",
    "Course",
    "Daytime/evening attendance",
    "Previous qualification",
]

FEATURE_LABELS = {
    "Age at enrollment": "입학 나이",
    "Curricular units 1st sem (approved)": "1학기 이수 과목 수",
    "Curricular units 1st sem (grade)": "1학기 평균 성적",
    "Curricular units 2nd sem (approved)": "2학기 이수 과목 수",
    "Curricular units 2nd sem (grade)": "2학기 평균 성적",
    "Tuition fees up to date": "학비 납부 여부",
    "Scholarship holder": "장학금 수혜",
    "Debtor": "채무 여부",
    "Gender": "성별 (1=남)",
    "Displaced": "타지역 거주",
    "Educational special needs": "특수 교육 필요",
    "International": "유학생 여부",
    "Unemployment rate": "실업률",
    "Inflation rate": "물가상승률",
    "GDP": "GDP",
    "Application mode": "지원 방식",
    "Course": "전공 코드",
    "Daytime/evening attendance": "주간/야간 (1=주간)",
    "Previous qualification": "이전 학력",
}


def prepare_xy(df: pd.DataFrame):
    """학습용 X, y 반환 (Enrolled 제외 → 이진 분류)"""
    df_filtered = df[df["Target"].isin(["Dropout", "Graduate"])].copy()
    X = df_filtered[KEY_FEATURES].fillna(0)
    y = df_filtered["Target"].map({"Dropout": 0, "Graduate": 1})
    return X, y


def prepare_all_xy(df: pd.DataFrame):
    """3-class 분류용 X, y"""
    X = df[KEY_FEATURES].fillna(0)
    y = df["Target"].map(TARGET_MAP)
    return X, y
