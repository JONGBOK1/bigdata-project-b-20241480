import streamlit as st
import pandas as pd
import plotly.express as px
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from src.data_loader import load_data

COLUMN_LABELS = {
    "Marital status": "혼인 상태",
    "Application mode": "지원 방식",
    "Application order": "지원 순서",
    "Course": "전공 코드",
    "Daytime/evening attendance": "주간/야간",
    "Previous qualification": "이전 학력",
    "Previous qualification (grade)": "이전 학력 성적",
    "Nationality": "국적",
    "Mother's qualification": "모친 학력",
    "Father's qualification": "부친 학력",
    "Mother's occupation": "모친 직업",
    "Father's occupation": "부친 직업",
    "Displaced": "타지역 거주",
    "Educational special needs": "특수 교육 필요",
    "Debtor": "채무 여부",
    "Tuition fees up to date": "학비 납부 여부",
    "Gender": "성별",
    "Scholarship holder": "장학금 수혜",
    "Age at enrollment": "입학 나이",
    "International": "유학생 여부",
    "Curricular units 1st sem (credited)": "1학기 인정 과목 수",
    "Curricular units 1st sem (enrolled)": "1학기 수강 신청 수",
    "Curricular units 1st sem (evaluations)": "1학기 평가 횟수",
    "Curricular units 1st sem (approved)": "1학기 이수 과목 수",
    "Curricular units 1st sem (grade)": "1학기 평균 성적",
    "Curricular units 1st sem (without evaluations)": "1학기 미평가 과목 수",
    "Curricular units 2nd sem (credited)": "2학기 인정 과목 수",
    "Curricular units 2nd sem (enrolled)": "2학기 수강 신청 수",
    "Curricular units 2nd sem (evaluations)": "2학기 평가 횟수",
    "Curricular units 2nd sem (approved)": "2학기 이수 과목 수",
    "Curricular units 2nd sem (grade)": "2학기 평균 성적",
    "Curricular units 2nd sem (without evaluations)": "2학기 미평가 과목 수",
    "Unemployment rate": "실업률",
    "Inflation rate": "물가상승률",
    "GDP": "GDP 성장률",
    "Target": "결과",
}

st.title("📊 EDA — 탐색적 데이터 분석")

df = load_data()
df_kor = df.rename(columns=COLUMN_LABELS)

# ── 1. 기본 정보 ──────────────────────────────────────────────
st.header("1. 데이터 기본 정보")
col1, col2, col3 = st.columns(3)
col1.metric("총 행 수", f"{len(df):,}")
col2.metric("총 열 수", f"{df.shape[1]}")
col3.metric("타겟 클래스 수", df["Target"].nunique())

st.subheader("상위 5행")
st.dataframe(df_kor.head())

st.subheader("컬럼별 데이터 타입")
st.dataframe(
    df_kor.dtypes.reset_index().rename(columns={"index": "컬럼", 0: "타입"})
)

# ── 2. 결측값 ──────────────────────────────────────────────────
st.header("2. 결측값 확인")
missing = df_kor.isnull().sum()
missing_pct = (missing / len(df_kor) * 100).round(2)
miss_df = pd.DataFrame({"결측 수": missing, "결측률(%)": missing_pct})
miss_df = miss_df[miss_df["결측 수"] > 0]

if miss_df.empty:
    st.success("결측값 없음 ✅")
else:
    st.dataframe(miss_df)

# ── 3. 타겟 분포 ───────────────────────────────────────────────
st.header("3. 타겟(Target) 분포")
target_counts = df["Target"].value_counts()
st.bar_chart(target_counts)
st.dataframe(
    target_counts.reset_index().rename(columns={"Target": "클래스", "count": "학생 수"})
)

# ── 4. 수치형 변수 기술통계 ───────────────────────────────────
st.header("4. 수치형 변수 기술통계")
st.dataframe(df_kor.describe().T)

# ── 5. 개별 컬럼 분포 ────────────────────────────────────────
st.header("5. 특정 컬럼 분포 확인")
num_cols = df.select_dtypes("number").columns.tolist()
kor_options = [COLUMN_LABELS.get(c, c) for c in num_cols]
selected_kor = st.selectbox("컬럼 선택", kor_options)
selected_orig = num_cols[kor_options.index(selected_kor)]

vc = df[selected_orig].value_counts().sort_index().reset_index()
vc.columns = ["값", "학생 수"]
vc["비율(%)"] = (vc["학생 수"] / vc["학생 수"].sum() * 100).round(1)
vc["레이블"] = vc["학생 수"].astype(str) + "명 (" + vc["비율(%)"].astype(str) + "%)"
vc["값"] = vc["값"].astype(str)

fig = px.bar(
    vc,
    x="값",
    y="학생 수",
    text="레이블",
    title=f"{selected_kor} 분포",
    labels={"값": selected_kor, "학생 수": "학생 수"},
)
fig.update_traces(textposition="outside")
fig.update_layout(uniformtext_minsize=10)
st.plotly_chart(fig, use_container_width=True)

st.dataframe(vc[["값", "학생 수", "비율(%)"]], use_container_width=True)
