import streamlit as st

st.set_page_config(
    page_title="학생 중퇴 예측 서비스",
    page_icon="🎓",
    layout="wide",
)

st.title("🎓 학생 중퇴율 & 학업 성공률 예측")
st.markdown(
    """
    포르투갈 고등교육 학생 데이터를 기반으로,
    **중퇴(Dropout) / 졸업(Graduate)** 여부를 예측하는 Streamlit 앱입니다.

    ---
    ### 사이드바에서 페이지를 선택하세요
    | 페이지 | 내용 |
    |--------|------|
    | 📊 1_EDA | 데이터 요약 · 결측 · 분포 확인 |
    | 📈 2_시각화 | 인사이트 그래프 |
    | 🤖 3_모델_서비스 | 입력 → 예측 → 결과 |
    """
)
