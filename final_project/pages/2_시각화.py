import streamlit as st
import pandas as pd
import plotly.express as px
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from src.data_loader import load_data

st.title("📈 시각화 — 인사이트 탐색")

df = load_data()

# ── 그래프 1: 타겟별 학비 납부 비율 ──────────────────────────────
st.header("1. 학비 납부 여부 × 결과 (중퇴 vs 졸업)")
st.markdown("""
**인사이트**: 학비를 납부하지 않은 학생은 중퇴율이 크게 높음.
학비 납부 여부는 중퇴 예측에서 매우 중요한 신호.
""")
fig1 = px.histogram(
    df,
    x="Target",
    color="Tuition fees up to date",
    barmode="group",
    labels={"Tuition fees up to date": "학비 납부 (1=납부)"},
    title="학비 납부 여부별 학업 결과 분포",
    color_discrete_map={0: "#EF553B", 1: "#00CC96"},
)
st.plotly_chart(fig1, use_container_width=True)

# ── 그래프 2: 1학기 성적 × 결과 박스플롯 ─────────────────────────
st.header("2. 1학기 평균 성적 분포 × 결과")
st.markdown("""
**인사이트**: Graduate(졸업) 학생의 1학기 성적 중앙값이 Dropout(중퇴)보다 유의미하게 높음.
초반 학업 성취도가 졸업 여부를 결정하는 핵심 요인.
""")
fig2 = px.box(
    df[df["Target"] != "Enrolled"],
    x="Target",
    y="Curricular units 1st sem (grade)",
    color="Target",
    title="1학기 평균 성적 분포 (중퇴 vs 졸업)",
    labels={"Curricular units 1st sem (grade)": "1학기 평균 성적"},
    color_discrete_map={"Dropout": "#EF553B", "Graduate": "#00CC96"},
)
st.plotly_chart(fig2, use_container_width=True)

# ── 그래프 3: 입학 나이 분포 ─────────────────────────────────────
st.header("3. 입학 나이 분포 × 결과")
st.markdown("""
**인사이트**: 입학 나이가 높을수록 중퇴 비율이 증가하는 경향.
만학도(30세 이상)에서 중퇴율이 두드러짐.
""")
fig3 = px.histogram(
    df[df["Target"] != "Enrolled"],
    x="Age at enrollment",
    color="Target",
    nbins=30,
    barmode="overlay",
    opacity=0.7,
    title="입학 나이 분포",
    labels={"Age at enrollment": "입학 나이"},
    color_discrete_map={"Dropout": "#EF553B", "Graduate": "#00CC96"},
)
st.plotly_chart(fig3, use_container_width=True)

# ── 그래프 4: 장학금 × 결과 ─────────────────────────────────────
st.header("4. 장학금 수혜 여부 × 결과")
st.markdown("""
**인사이트**: 장학금을 받는 학생의 졸업률이 현저히 높음.
재정 지원이 학업 지속에 핵심적 역할을 함.
""")
scholarship_target = (
    df.groupby(["Scholarship holder", "Target"])
    .size()
    .reset_index(name="count")
)
scholarship_target["장학금"] = scholarship_target["Scholarship holder"].map(
    {0: "미수혜", 1: "수혜"}
)
fig4 = px.bar(
    scholarship_target,
    x="장학금",
    y="count",
    color="Target",
    barmode="group",
    title="장학금 수혜 여부별 학업 결과",
    color_discrete_map={"Dropout": "#EF553B", "Enrolled": "#636EFA", "Graduate": "#00CC96"},
)
st.plotly_chart(fig4, use_container_width=True)
