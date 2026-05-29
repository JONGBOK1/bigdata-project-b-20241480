import streamlit as st
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from src.data_loader import load_data

st.title("📊 EDA — 탐색적 데이터 분석")

df = load_data()

# ── 1. 기본 정보 ──────────────────────────────────────────────
st.header("1. 데이터 기본 정보")
col1, col2, col3 = st.columns(3)
col1.metric("총 행 수", f"{len(df):,}")
col2.metric("총 열 수", f"{df.shape[1]}")
col3.metric("타겟 클래스 수", df["Target"].nunique())

st.subheader("상위 5행")
st.dataframe(df.head())

st.subheader("컬럼별 데이터 타입")
st.dataframe(
    df.dtypes.reset_index().rename(columns={"index": "컬럼", 0: "타입"})
)

# ── 2. 결측값 ──────────────────────────────────────────────────
st.header("2. 결측값 확인")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
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
    target_counts.reset_index().rename(columns={"index": "클래스", "Target": "학생 수"})
)

# ── 4. 수치형 변수 기술통계 ───────────────────────────────────
st.header("4. 수치형 변수 기술통계")
st.dataframe(df.describe().T)

# ── 5. 개별 컬럼 분포 ────────────────────────────────────────
st.header("5. 특정 컬럼 분포 확인")
col_select = st.selectbox("컬럼 선택", df.select_dtypes("number").columns.tolist())
st.bar_chart(df[col_select].value_counts().sort_index())
