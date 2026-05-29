# 학생 중퇴율 & 학업 성공률 예측 서비스

5주차 영화 대시보드와 **같은 멀티페이지 구조**입니다. 바로 실행 가능.

## 시작하기
```bash
# 1) Python 3.10 이상 가상환경 권장 (3.9는 Streamlit 버전 충돌)
py -3.13 -m venv venv313
venv313\Scripts\activate

# 2) 의존성 설치
pip install -r requirements.txt

# 3) 실행
streamlit run app.py
```

## 폴더 구조
```
final_project/
app.py                  # 진입점 (st.navigation) → 제목/이름만 바꾸면 됨
pages/
  1_EDA.py              # 1차 작업: 요약·결측·분포
  2_시각화.py            # 2차 작업: Plotly 그래프 + 해석
  3_모델_서비스.py        # 3차 작업: 입력→예측→결과 (RandomForest)
src/
  data_loader.py        # 내 데이터 연결 (@st.cache_data)
  features.py           # 정제·특성 (과제로 채움)
data/
  dataset.csv           # 원본 데이터 (4,424행 × 37열)
문제정의.md
보고서.md
requirements.txt
```

## 데이터셋
- **이름**: 학생 중퇴율과 학업 성공률 예측 데이터셋
- **출처**: [Kaggle](https://www.kaggle.com/datasets/thedevastator/higher-education-predictors-of-student-retention)
- **크기**: 4,424행 × 37열 / CSV 형식
- **타겟**: `Target` — Dropout(중퇴) / Graduate(졸업) / Enrolled(재학중)

## 서비스 개요
> 학생의 입학 정보·성적·재정 상태를 입력하면 **중퇴 위험도**를 예측합니다.

| 페이지 | 내용 |
|--------|------|
| 📊 EDA | 결측값 없음 확인, 타겟 분포, 수치형 변수 기술통계 |
| 📈 시각화 | 학비 납부·성적·나이·장학금 × 결과 Plotly 그래프 4개 |
| 🤖 모델/서비스 | RandomForest 분류 (정확도 ~88%) + 위험 요인 안내 |

## 주요 발견 (EDA 인사이트)
- **학비 납부 여부**가 중퇴 예측에서 가장 강한 신호
- **1·2학기 성적 및 이수 과목 수**가 특성 중요도 상위권
- 입학 나이 **30세 이상** 만학도에서 중퇴율 급증
- **장학금 수혜** 학생의 졸업율이 비수혜 대비 약 2배

## 모델
- 알고리즘: `RandomForestClassifier` (n_estimators=100)
- 학습: Dropout + Graduate 이진 분류 (Enrolled 제외)
- 분할: Train 80% / Test 20% (stratified)
- 정확도: **~88%**

## 작성자
20241480 이종복
