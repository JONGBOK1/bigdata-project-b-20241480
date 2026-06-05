import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, balanced_accuracy_score
import plotly.express as px
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from src.data_loader import load_data
from src.features import prepare_xy, KEY_FEATURES, FEATURE_LABELS

st.title("🤖 모델/서비스 — 중퇴 예측")

# ── 모델 학습 (캐시) ──────────────────────────────────────────────
@st.cache_resource
def train_model():
    df = load_data()
    X, y = prepare_xy(df)  # Enrolled 제외, 이진 분류
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    bal_acc = balanced_accuracy_score(y_te, y_pred)
    report = classification_report(y_te, y_pred,
                                   target_names=["중퇴", "졸업"], output_dict=True)
    return clf, acc, bal_acc, report, X_tr.columns.tolist()

clf, acc, bal_acc, report, feature_cols = train_model()

# ── 모델 성능 요약 ───────────────────────────────────────────────
st.header("1. 모델 성능 (테스트셋)")
col1, col2, col3, col4 = st.columns(4)
col1.metric("정확도", f"{acc*100:.1f}%")
col2.metric("균형 정확도", f"{bal_acc*100:.1f}%")
col3.metric("중퇴 F1-score", f"{report['중퇴']['f1-score']*100:.1f}%")
col4.metric("졸업 F1-score", f"{report['졸업']['f1-score']*100:.1f}%")

# 특성 중요도
st.subheader("특성 중요도 (상위 10)")
importances = pd.Series(clf.feature_importances_, index=feature_cols)
top10 = importances.nlargest(10).reset_index()
top10.columns = ["특성", "중요도"]
top10["특성_kor"] = top10["특성"].map(FEATURE_LABELS).fillna(top10["특성"])
fig_imp = px.bar(
    top10.sort_values("중요도"),
    x="중요도",
    y="특성_kor",
    orientation="h",
    title="Random Forest 특성 중요도",
)
st.plotly_chart(fig_imp, use_container_width=True)

st.divider()

# ── 서비스: 사용자 입력 → 예측 ──────────────────────────────────
st.header("2. 학생 정보 입력 → 중퇴 위험 예측")
st.markdown("아래 정보를 입력하면 중퇴 위험도를 예측합니다.")

with st.form("predict_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("입학 나이", min_value=17, max_value=70, value=20)
        sem1_approved = st.number_input("1학기 이수 과목 수", 0, 20, 5)
        sem1_grade = st.number_input("1학기 평균 성적", 0.0, 20.0, 12.0, step=0.5)
        sem2_approved = st.number_input("2학기 이수 과목 수", 0, 20, 5)
        sem2_grade = st.number_input("2학기 평균 성적", 0.0, 20.0, 12.0, step=0.5)
        course = st.number_input("전공 코드", 1, 20, 9)

    with col2:
        tuition = st.selectbox("학비 납부 여부", [1, 0], format_func=lambda x: "납부" if x == 1 else "미납")
        scholarship = st.selectbox("장학금 수혜", [0, 1], format_func=lambda x: "수혜" if x == 1 else "미수혜")
        debtor = st.selectbox("채무 여부", [0, 1], format_func=lambda x: "있음" if x == 1 else "없음")
        gender = st.selectbox("성별", [1, 0], format_func=lambda x: "남" if x == 1 else "여")
        displaced = st.selectbox("타지역 거주", [0, 1], format_func=lambda x: "예" if x == 1 else "아니오")

    with col3:
        special_needs = st.selectbox("특수 교육 필요", [0, 1], format_func=lambda x: "예" if x == 1 else "아니오")
        international = st.selectbox("유학생 여부", [0, 1], format_func=lambda x: "예" if x == 1 else "아니오")
        unemployment = st.number_input("실업률 (%)", 5.0, 20.0, 11.0, step=0.1)
        inflation = st.number_input("물가상승률 (%)", -1.0, 5.0, 1.4, step=0.1)
        gdp = st.number_input("GDP 성장률", -5.0, 5.0, 1.7, step=0.1)
        app_mode = st.number_input("지원 방식 코드", 1, 20, 1)
        daytime = st.selectbox("주간/야간", [1, 0], format_func=lambda x: "주간" if x == 1 else "야간")
        prev_qual = st.number_input("이전 학력 코드", 1, 40, 1)

    submitted = st.form_submit_button("🔍 예측하기", use_container_width=True)

if submitted:
    input_data = pd.DataFrame([{
        "Age at enrollment": age,
        "Curricular units 1st sem (approved)": sem1_approved,
        "Curricular units 1st sem (grade)": sem1_grade,
        "Curricular units 2nd sem (approved)": sem2_approved,
        "Curricular units 2nd sem (grade)": sem2_grade,
        "Tuition fees up to date": tuition,
        "Scholarship holder": scholarship,
        "Debtor": debtor,
        "Gender": gender,
        "Displaced": displaced,
        "Educational special needs": special_needs,
        "International": international,
        "Unemployment rate": unemployment,
        "Inflation rate": inflation,
        "GDP": gdp,
        "Application mode": app_mode,
        "Course": course,
        "Daytime/evening attendance": daytime,
        "Previous qualification": prev_qual,
    }])

    pred = clf.predict(input_data)[0]
    proba = clf.predict_proba(input_data)[0]
    dropout_prob = proba[0] * 100
    graduate_prob = proba[1] * 100

    st.divider()
    st.subheader("📋 예측 결과")

    if pred == 0:
        st.error(f"⚠️ **중퇴 위험** — 중퇴 확률: {dropout_prob:.1f}%")
    else:
        st.success(f"🎓 **졸업 예상** — 졸업 확률: {graduate_prob:.1f}%")

    # 확률 게이지
    prob_df = pd.DataFrame({
        "결과": ["중퇴", "졸업"],
        "확률(%)": [dropout_prob, graduate_prob],
    })
    fig_prob = px.bar(
        prob_df,
        x="결과",
        y="확률(%)",
        color="결과",
        color_discrete_map={"중퇴": "#EF553B", "졸업": "#00CC96"},
        title="예측 확률",
        range_y=[0, 100],
    )
    st.plotly_chart(fig_prob, use_container_width=True)

    # 위험 요인 안내
    st.subheader("💡 주요 위험 요인 점검")
    warnings = []
    if tuition == 0:
        warnings.append("학비 미납 상태입니다. 납부 여부가 중퇴율에 큰 영향을 줍니다.")
    if sem1_approved < 3:
        warnings.append("1학기 이수 과목 수가 적습니다. 학업 참여도를 높이세요.")
    if sem1_grade < 10:
        warnings.append("1학기 성적이 낮습니다. 학습 지원이 필요할 수 있습니다.")
    if debtor == 1:
        warnings.append("채무가 있습니다. 재정 지원 상담을 고려하세요.")
    if age > 30:
        warnings.append("만학도 학생입니다. 추가적인 학업 지원이 필요할 수 있습니다.")

    if warnings:
        for w in warnings:
            st.warning(f"• {w}")
    else:
        st.info("특별한 위험 요인이 발견되지 않았습니다.")
