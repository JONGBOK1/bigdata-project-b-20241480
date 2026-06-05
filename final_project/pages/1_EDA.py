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
    "Course": "전공",
    "Daytime/evening attendance": "주간/야간",
    "Previous qualification": "이전 학력",
    "Previous qualification (grade)": "이전 학력 성적",
    "Nacionality": "국적",
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

VALUE_LABELS = {
    "Marital status": {
        1: "미혼", 2: "기혼", 3: "사별", 4: "이혼", 5: "사실혼", 6: "법적 별거",
    },
    "Application mode": {
        1: "1차 일반 전형",
        2: "특별 법령(612/93)",
        3: "1차 특별 전형(아조레스)",
        4: "타 고등과정 이수자",
        5: "특별 법령(854-B)",
        6: "유학생(학사)",
        7: "1차 특별 전형(마데이라)",
        8: "2차 일반 전형",
        9: "3차 일반 전형",
        10: "특별 법령(533-A, 타 계획)",
        11: "특별 법령(533-A, 타 기관)",
        12: "23세 이상",
        13: "편입",
        14: "전과",
        15: "기술전문 졸업자",
        16: "기관·학과 변경",
        17: "단기과정 졸업자",
        18: "기관·학과 변경(유학생)",
    },
    "Course": {
        1:  "바이오연료생산기술",
        2:  "애니메이션·멀티미디어디자인",
        3:  "사회복지(야간)",
        4:  "농학",
        5:  "커뮤니케이션디자인",
        6:  "수의간호학",
        7:  "정보공학",
        8:  "승마학",
        9:  "경영학",
        10: "사회복지",
        11: "관광학",
        12: "간호학",
        13: "구강위생학",
        14: "광고·마케팅경영학",
        15: "저널리즘·커뮤니케이션",
        16: "기초교육학",
        17: "경영학(야간)",
    },
    "Daytime/evening attendance": {1: "주간", 0: "야간"},
    "Previous qualification": {
        1:  "고등학교 졸업",
        2:  "학사 학위 보유",
        3:  "고등교육 학위",
        4:  "석사",
        5:  "박사",
        6:  "고등교육 수강 중",
        7:  "12학년 미수료",
        8:  "11학년 미수료",
        9:  "11학년(기타)",
        10: "10학년",
        11: "10학년 미수료",
        12: "중등교육 3주기(9~11학년)",
        13: "중등교육 2주기(6~8학년)",
        14: "기술전문 과정",
        15: "고등교육 학위(1주기)",
        16: "전문기술 고등과정",
        17: "석사(2주기)",
    },
    "Nacionality": {
        1:  "포르투갈", 2:  "독일",   3:  "스페인",   4:  "이탈리아",
        5:  "네덜란드", 6:  "영국",   7:  "리투아니아", 8:  "앙골라",
        9:  "카보베르데", 10: "기니",  11: "모잠비크",  12: "상투메",
        13: "터키",    14: "브라질", 15: "루마니아",  16: "몰도바",
        17: "멕시코",  18: "우크라이나", 19: "러시아", 20: "쿠바",
        21: "콜롬비아",
    },
    "Mother's qualification": {
        1:  "고교 졸업(12학년)", 2:  "학사",         3:  "학위(3년제)",
        4:  "석사",             5:  "박사",         6:  "고등교육 재학",
        7:  "12학년 미수료",    8:  "11학년 미수료", 9:  "7학년(구과정)",
        10: "11학년(기타)",     11: "고교 보완과정 2년차", 12: "10학년",
        13: "상업 일반과정",    14: "중학교(9~11학년)", 15: "고교 보완과정",
        16: "기술직업 과정",    17: "고교 보완과정(미수료)", 18: "7학년",
        19: "고교 2주기",       20: "9학년 미수료", 21: "8학년",
        22: "행정·상업 일반과정", 23: "회계·행정 보충과정", 24: "불명",
        25: "문맹",             26: "4학년 미만 독해 가능", 27: "초등학교(4~5학년)",
        28: "중학교(6~8학년)",  29: "기술전문 과정",
    },
    "Father's qualification": {
        1:  "고교 졸업(12학년)", 2:  "학사",         3:  "학위(3년제)",
        4:  "석사",             5:  "박사",         6:  "고등교육 재학",
        7:  "12학년 미수료",    8:  "11학년 미수료", 9:  "7학년(구과정)",
        10: "11학년(기타)",     11: "고교 보완과정 2년차", 12: "10학년",
        13: "상업 일반과정",    14: "중학교(9~11학년)", 15: "고교 보완과정",
        16: "기술직업 과정",    17: "고교 보완과정(미수료)", 18: "7학년",
        19: "고교 2주기",       20: "9학년 미수료", 21: "8학년",
        22: "행정·상업 일반과정", 23: "회계·행정 보충과정", 24: "불명",
        25: "문맹",             26: "4학년 미만 독해 가능", 27: "초등학교(4~5학년)",
        28: "중학교(6~8학년)",  29: "기술전문 과정",
        30: "고등교육 학위(1주기)", 31: "전문고등과정",
        32: "전문기술 고등과정", 33: "석사(2주기)", 34: "박사(3주기)",
    },
    "Mother's occupation": {
        1:  "학생",           2:  "관리·임원직",    3:  "지식·과학 전문직",
        4:  "중간기술직",     5:  "사무직",         6:  "서비스·판매직",
        7:  "농업·어업 숙련직", 8: "건설·제조 숙련직", 9:  "기계·조립 운전직",
        10: "단순 노무직",    11: "군인",           12: "기타",
        13: "미상",           14: "의료 전문직",    15: "교사",
        16: "ICT 전문직",     17: "이공계 중간기술직", 18: "보건 중간기술직",
        19: "법·사회·스포츠 중간기술직", 20: "사무·데이터 처리",
        21: "회계·통계·금융 사무", 22: "기타 행정 지원", 23: "개인 서비스직",
        24: "판매원",         25: "돌봄 서비스",    26: "건설 숙련직",
        27: "정밀기계·귀금속 숙련직", 28: "식품·목공·의류 숙련직",
        29: "청소 노무직",    30: "농업·어업 단순 노무직",
        31: "건설·제조·운반 단순 노무직", 32: "식음료 준비 보조",
    },
    "Father's occupation": {
        1:  "학생",           2:  "관리·임원직",    3:  "지식·과학 전문직",
        4:  "중간기술직",     5:  "사무직",         6:  "서비스·판매직",
        7:  "농업·어업 숙련직", 8: "건설·제조 숙련직", 9:  "기계·조립 운전직",
        10: "단순 노무직",    11: "군인",           12: "기타",
        13: "미상",           14: "의료 전문직",    15: "교사",
        16: "ICT 전문직",     17: "이공계 중간기술직", 18: "보건 중간기술직",
        19: "법·사회·스포츠 중간기술직", 20: "사무·데이터 처리",
        21: "회계·통계·금융 사무", 22: "기타 행정 지원", 23: "개인 서비스직",
        24: "판매원",         25: "돌봄 서비스",    26: "건설 숙련직",
        27: "정밀기계·귀금속 숙련직", 28: "식품·목공·의류 숙련직",
        29: "청소 노무직",    30: "농업·어업 단순 노무직",
        31: "건설·제조·운반 단순 노무직", 32: "식음료 준비 보조",
        33: "서비스직(기타)", 34: "농업 관련직",    35: "어업 관련직",
        36: "목공 숙련직",    37: "금속 숙련직",    38: "전기·전자 숙련직",
        39: "식품가공 숙련직", 40: "의류·가죽 숙련직", 41: "제지·인쇄 숙련직",
        42: "고무·플라스틱 숙련직", 43: "운전직",  44: "하역·창고직",
        45: "건설 단순 노무직", 46: "기타 단순 노무직",
    },
    "Application order": {
        0: "미지정", 1: "1지망", 2: "2지망", 3: "3지망",
        4: "4지망", 5: "5지망", 6: "6지망", 9: "9지망",
    },
    "Displaced": {0: "아니오", 1: "예"},
    "Educational special needs": {0: "아니오", 1: "예"},
    "Debtor": {0: "없음", 1: "있음"},
    "Tuition fees up to date": {0: "미납", 1: "납부"},
    "Gender": {0: "여성", 1: "남성"},
    "Scholarship holder": {0: "미수혜", 1: "수혜"},
    "International": {0: "아니오", 1: "예"},
    "Unemployment rate": {
        7.6: "7.6% (낮음)", 8.9: "8.9% (낮음)",
        9.4: "9.4% (낮음)", 10.8: "10.8% (보통)",
        11.1: "11.1% (보통)", 12.4: "12.4% (보통)",
        12.7: "12.7% (높음)", 13.9: "13.9% (높음)",
        15.5: "15.5% (매우 높음)", 16.2: "16.2% (매우 높음)",
    },
    "Inflation rate": {
        -0.8: "-0.8% (디플레이션)", -0.3: "-0.3% (디플레이션)",
         0.3: "0.3% (안정)",  0.5: "0.5% (안정)",
         0.6: "0.6% (안정)",  1.4: "1.4% (안정)",
         2.6: "2.6% (상승)",  2.8: "2.8% (상승)",
         3.7: "3.7% (고물가)",
    },
    "GDP": {
        -4.06: "-4.06% (심각한 침체)", -3.12: "-3.12% (침체)",
        -1.7:  "-1.70% (침체)",       -0.92: "-0.92% (침체)",
         0.32: "0.32% (저성장)",       0.79: "0.79% (저성장)",
         1.74: "1.74% (성장)",         1.79: "1.79% (성장)",
         2.02: "2.02% (성장)",         3.51: "3.51% (고성장)",
    },
}

st.title("📊 EDA — 탐색적 데이터 분석")

df = load_data()
df_kor = df.rename(columns=COLUMN_LABELS)

# ── 0. 원본 데이터 ─────────────────────────────────────────────
with st.expander("📂 원본 데이터 보기 (영문 컬럼·숫자 그대로)", expanded=False):
    st.caption(f"전체 {len(df):,}행 × {df.shape[1]}열")
    col_filter = st.multiselect(
        "표시할 컬럼 선택 (미선택 시 전체 표시)",
        options=df.columns.tolist(),
        default=[],
        key="raw_col_filter",
    )
    show_cols = col_filter if col_filter else df.columns.tolist()
    st.dataframe(df[show_cols], use_container_width=True, height=400)

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

val_map = VALUE_LABELS.get(selected_orig, {})
if val_map:
    vc["x축"] = vc["값"].map(lambda v: val_map.get(v, str(v)))
else:
    vc["x축"] = vc["값"].astype(str)

vc["레이블"] = vc["학생 수"].astype(str) + "명 (" + vc["비율(%)"].astype(str) + "%)"

fig = px.bar(
    vc,
    x="x축",
    y="학생 수",
    text="레이블",
    title=f"{selected_kor} 분포",
    labels={"x축": selected_kor, "학생 수": "학생 수"},
)
fig.update_traces(textposition="outside")
fig.update_layout(uniformtext_minsize=10, xaxis_tickangle=-30)
st.plotly_chart(fig, use_container_width=True)

display_vc = vc[["x축", "학생 수", "비율(%)"]].rename(columns={"x축": "항목"})
st.dataframe(display_vc, use_container_width=True)
