from docx import Document
from docx.shared import Pt, RGBColor, Inches, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from pathlib import Path

doc = Document()

# ── 기본 스타일 설정 ───────────────────────────────────────────
style = doc.styles['Normal']
style.font.name = '맑은 고딕'
style.font.size = Pt(11)

def set_font(run, size=11, bold=False, color=None):
    run.font.name = '맑은 고딕'
    run.font.size = Pt(size)
    run.font.bold = bold
    if color:
        run.font.color.rgb = RGBColor(*color)

def add_heading(doc, text, level=1):
    p = doc.add_heading(text, level=level)
    for run in p.runs:
        run.font.name = '맑은 고딕'
    return p

def add_para(doc, text, bold=False, size=11, color=None, indent=False):
    p = doc.add_paragraph()
    if indent:
        p.paragraph_format.left_indent = Cm(1)
    run = p.add_run(text)
    set_font(run, size=size, bold=bold, color=color)
    return p

def add_table(doc, headers, rows, col_widths=None):
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = 'Table Grid'
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    # 헤더
    for i, h in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.text = h
        run = cell.paragraphs[0].runs[0]
        set_font(run, bold=True, size=10)
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    # 데이터
    for r, row in enumerate(rows):
        for c, val in enumerate(row):
            cell = table.rows[r + 1].cells[c]
            cell.text = val
            run = cell.paragraphs[0].runs[0]
            set_font(run, size=10)
    if col_widths:
        for r in table.rows:
            for c, w in enumerate(col_widths):
                r.cells[c].width = Cm(w)
    return table

def add_image_placeholder(doc, caption):
    doc.add_paragraph()
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(f"[ 캡처 이미지 삽입: {caption} ]")
    run.font.name = '맑은 고딕'
    run.font.size = Pt(10)
    run.font.color.rgb = RGBColor(0x99, 0x99, 0x99)
    run.font.italic = True
    # 회색 박스 효과 (빈 줄)
    p2 = doc.add_paragraph()
    p2.paragraph_format.space_after = Pt(12)
    doc.add_paragraph()

# ══════════════════════════════════════════════════════════════
# 제목
# ══════════════════════════════════════════════════════════════
title = doc.add_heading('학생 중퇴 예측 AI 서비스', 0)
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
for run in title.runs:
    run.font.name = '맑은 고딕'

doc.add_paragraph()
info_lines = [
    ('이름 / 학번',   '20241480'),
    ('데이터 출처',   'Kaggle — Higher Education Predictors of Student Retention'),
    ('라이선스',     'CC BY 4.0'),
]
for label, val in info_lines:
    p = doc.add_paragraph()
    r1 = p.add_run(f'{label}: ')
    set_font(r1, bold=True)
    r2 = p.add_run(val)
    set_font(r2)

doc.add_page_break()

# ══════════════════════════════════════════════════════════════
# 1. 문제 정의
# ══════════════════════════════════════════════════════════════
add_heading(doc, '1. 문제 정의')

add_para(doc,
    '대학교에서 학생의 중퇴는 학생 본인은 물론 교육 기관에도 큰 손실이다. '
    '만약 중퇴 위험이 높은 학생을 사전에 파악할 수 있다면, 교수·상담사·학교 행정팀이 '
    '조기에 개입하여 중퇴를 예방할 수 있다. 본 프로젝트는 포르투갈 고등교육 학생 데이터를 '
    '활용해 학생이 중퇴할 가능성을 예측하는 웹 서비스를 구축하는 것을 목표로 한다.')

doc.add_paragraph()
for text in ['예측 대상(Target): 중퇴(Dropout) vs 졸업(Graduate) 이진 분류',
             '서비스 목적: 학생 정보를 입력하면 중퇴 위험 확률과 주요 위험 요인을 즉시 제공']:
    p = doc.add_paragraph(style='List Bullet')
    run = p.add_run(text)
    set_font(run)

add_heading(doc, '문제 재정의', level=2)
add_para(doc,
    '원래 데이터셋의 Target은 Dropout / Graduate / Enrolled 3개 클래스로 구성된다. '
    '그러나 Enrolled(재학중)는 아직 결과가 확정되지 않은 학생으로, 학습 데이터로 사용할 경우 '
    '라벨 노이즈가 발생한다. 이에 Enrolled를 제외하고 Dropout vs Graduate 이진 분류 문제로 재정의하였다.')

doc.add_paragraph()
add_table(doc,
    ['구분', '학생 수', '비율'],
    [
        ['Graduate (졸업)', '2,209명', '49.9%'],
        ['Dropout (중퇴)',  '1,421명', '32.1%'],
        ['Enrolled (재학중, 제외)', '794명', '17.9%'],
        ['학습 데이터 합계', '3,630명', '-'],
    ],
    col_widths=[7, 4, 3])

# ══════════════════════════════════════════════════════════════
# 2. 데이터 소개 & EDA
# ══════════════════════════════════════════════════════════════
doc.add_paragraph()
add_heading(doc, '2. 데이터 소개 & EDA')

add_heading(doc, '데이터 기본 정보', level=2)
add_table(doc,
    ['항목', '내용'],
    [
        ['행 수', '4,424행'],
        ['열 수', '37열'],
        ['형식', 'CSV'],
        ['결측값', '없음'],
        ['수치형 변수', '36개'],
        ['범주형 변수', '1개 (Target)'],
    ],
    col_widths=[5, 9])

add_heading(doc, 'EDA 주요 발견', level=2)

add_heading(doc, '발견 1 — 1학기 성적 차이', level=3)
add_para(doc, '중퇴 학생과 졸업 학생의 1학기 평균 성적을 비교한 결과 뚜렷한 차이가 확인됐다.')
doc.add_paragraph()
add_table(doc,
    ['그룹', '1학기 평균 성적'],
    [
        ['중퇴 (Dropout)', '7.3점'],
        ['졸업 (Graduate)', '12.6점'],
    ],
    col_widths=[7, 7])
add_para(doc, '→ 졸업 학생이 중퇴 학생보다 약 1.7배 높은 성적. 입학 첫 학기 성적이 중퇴 위험의 조기 신호.')
add_image_placeholder(doc, '1학기 성적 분포 비교 그래프')

add_heading(doc, '발견 2 — 학비 납부 여부', level=3)
doc.add_paragraph()
add_table(doc,
    ['그룹', '학비 미납 비율'],
    [
        ['중퇴 (Dropout)', '32.2%'],
        ['졸업 (Graduate)', '1.3%'],
    ],
    col_widths=[7, 7])
add_para(doc, '→ 중퇴 학생의 학비 미납 비율이 졸업 학생 대비 약 25배 높음. 재정 지원이 중퇴 예방의 핵심.')
add_image_placeholder(doc, '학비 납부 여부 분포 그래프')

add_heading(doc, '발견 3 — 결측값 없음', level=3)
add_para(doc, '전체 37개 컬럼 모두 결측값이 존재하지 않아 별도의 결측치 처리는 수행하지 않았다.')
add_para(doc, '학습 전처리 단계에서 fillna(0)을 적용하였다.')

# ══════════════════════════════════════════════════════════════
# 3. 특성 엔지니어링
# ══════════════════════════════════════════════════════════════
doc.add_paragraph()
add_heading(doc, '3. 특성 엔지니어링')
add_para(doc,
    '37개 전체 변수 중 모델 해석 가능성과 실제 서비스 활용성이 높은 19개 변수를 선별하였다.')

add_heading(doc, '선택한 주요 특성 (19개)', level=2)
add_table(doc,
    ['변수명', '한국어 설명', '선택 근거'],
    [
        ['Curricular units 2nd sem (approved)', '2학기 이수 과목 수', '중요도 1위 (23.1%)'],
        ['Curricular units 2nd sem (grade)',    '2학기 평균 성적',    '중요도 2위 (15.1%)'],
        ['Curricular units 1st sem (approved)', '1학기 이수 과목 수', '중요도 3위 (13.9%)'],
        ['Curricular units 1st sem (grade)',    '1학기 평균 성적',    '중요도 4위 (12.1%)'],
        ['Tuition fees up to date',             '학비 납부 여부',     '중요도 5위, EDA 핵심 발견'],
        ['Course',                              '전공',              '전공별 중퇴율 차이 존재'],
        ['Age at enrollment',                   '입학 나이',          '만학도 여부와 중퇴 연관'],
        ['Scholarship holder',                  '장학금 수혜 여부',   '재정 상태 반영'],
        ['Debtor',                              '채무 여부',          '재정 상태 반영'],
        ['Gender',                              '성별',              '인구통계 변수'],
        ['Displaced',                           '타지역 거주 여부',   '생활 여건 반영'],
        ['Unemployment rate',                   '실업률',            '거시경제 환경'],
        ['Inflation rate',                      '물가상승률',         '거시경제 환경'],
        ['GDP',                                 'GDP 성장률',        '거시경제 환경'],
        ['Application mode',                    '지원 방식',          '입학 경로 차이'],
        ['Daytime/evening attendance',          '주간/야간',          '학습 환경 차이'],
        ['Previous qualification',              '이전 학력',          '학업 배경'],
        ['Educational special needs',           '특수 교육 필요 여부', '학습 지원 필요 여부'],
        ['International',                       '유학생 여부',        '환경적 취약성'],
    ],
    col_widths=[6, 4, 4])

add_heading(doc, '제외한 변수 및 이유', level=2)
add_table(doc,
    ['제외 변수', '이유'],
    [
        ['Curricular units (credited) 1·2학기',        '90% 이상이 0 → 분산 없음'],
        ['Curricular units (without evaluations) 1·2학기', '93% 이상이 0 → 분산 없음'],
        ['Curricular units (enrolled/evaluations)',    '이수 과목 수와 중복 정보'],
        ['Application order',                          '중퇴와 직접 연관성 낮음'],
        ['Nacionality',                                '대부분 포르투갈(88%) → 분산 낮음'],
        ["Mother's / Father's qualification·occupation", '입력 부담 크고 중요도 낮음'],
        ['Previous qualification (grade)',             '현재 성적 변수와 중복'],
    ],
    col_widths=[7, 7])

# ══════════════════════════════════════════════════════════════
# 4. 모델 / 서비스
# ══════════════════════════════════════════════════════════════
doc.add_paragraph()
add_heading(doc, '4. 모델 / 서비스')

add_heading(doc, '사용 알고리즘', level=2)
for text in [
    'Random Forest Classifier (scikit-learn)',
    'n_estimators=100, random_state=42, n_jobs=-1',
    '선택 이유: 비선형 관계와 변수 간 상호작용을 잘 포착하며, 학생 중퇴 예측 관련 연구에서 높은 성능이 검증된 알고리즘',
    '학습/테스트 비율: 80% / 20% (stratified split)',
]:
    p = doc.add_paragraph(style='List Bullet')
    run = p.add_run(text)
    set_font(run)

add_heading(doc, '입력 → 출력 흐름', level=2)
for step in [
    '① 사용자 입력 (19개 변수)',
    '② 결측값 처리 (fillna(0))',
    '③ Random Forest 예측',
    '④ 중퇴 확률 / 졸업 확률 출력',
    '⑤ 주요 위험 요인 안내',
]:
    p = doc.add_paragraph(style='List Number')
    run = p.add_run(step)
    set_font(run)

add_heading(doc, '성능 지표', level=2)
add_table(doc,
    ['지표', '값'],
    [
        ['정확도 (Accuracy)',              '89.9%'],
        ['균형 정확도 (Balanced Accuracy)', '88.5%'],
        ['중퇴 F1-score',                  '86.0%'],
        ['졸업 F1-score',                  '92.0%'],
    ],
    col_widths=[8, 6])

doc.add_paragraph()
add_para(doc,
    '※ 균형 정확도를 추가한 이유: 중퇴(1,421명) vs 졸업(2,209명)의 클래스 불균형이 존재하여 '
    '단순 정확도만으로는 다수 클래스에 편향될 수 있다. 균형 정확도는 각 클래스 정확도의 '
    '평균으로 불균형 데이터에서 더 신뢰할 수 있는 지표이다.',
    color=(100, 100, 100))

add_heading(doc, '특성 중요도 TOP 5', level=2)
add_table(doc,
    ['순위', '변수', '중요도'],
    [
        ['1', '2학기 이수 과목 수', '23.1%'],
        ['2', '2학기 평균 성적',   '15.1%'],
        ['3', '1학기 이수 과목 수', '13.9%'],
        ['4', '1학기 평균 성적',   '12.1%'],
        ['5', '학비 납부 여부',    '6.7%'],
    ],
    col_widths=[2, 9, 3])
add_image_placeholder(doc, '특성 중요도 막대 그래프')

# ══════════════════════════════════════════════════════════════
# 5. Streamlit 앱
# ══════════════════════════════════════════════════════════════
doc.add_paragraph()
add_heading(doc, '5. Streamlit 앱')

add_heading(doc, '페이지 구성', level=2)
add_table(doc,
    ['페이지', '내용'],
    [
        ['app.py',               '메인 홈 화면'],
        ['pages/1_EDA.py',       '탐색적 데이터 분석 — 기본 정보, 결측값 확인, 타겟 분포, 컬럼별 분포 시각화'],
        ['pages/2_시각화.py',    '주요 변수 시각화 — 중퇴/졸업 그룹 비교 그래프'],
        ['pages/3_모델_서비스.py', '모델 성능 확인 + 학생 정보 입력 → 중퇴 위험 예측 서비스'],
    ],
    col_widths=[5, 9])

doc.add_paragraph()
add_image_placeholder(doc, 'EDA 페이지 화면 캡처')
add_image_placeholder(doc, '시각화 페이지 화면 캡처')
add_image_placeholder(doc, '모델 서비스 페이지 — 입력 폼 캡처')
add_image_placeholder(doc, '모델 서비스 페이지 — 예측 결과 캡처')

add_heading(doc, '실행 방법', level=2)
for step in [
    'pip install -r requirements.txt',
    'streamlit run app.py',
]:
    p = doc.add_paragraph(style='List Number')
    run = p.add_run(step)
    set_font(run)
    run.font.name = 'Courier New'

# ══════════════════════════════════════════════════════════════
# 6. 결론 & 한계
# ══════════════════════════════════════════════════════════════
doc.add_paragraph()
add_heading(doc, '6. 결론 & 한계')

add_heading(doc, '알게 된 것', level=2)
for text in [
    '포르투갈 고등교육 학생 데이터만으로 중퇴 위험을 정확도 89.9%로 예측할 수 있었다.',
    '성적과 이수 과목 수(1·2학기)가 중퇴 예측에 가장 결정적인 요인이었다.',
    '학비 납부 여부가 성적 다음으로 강력한 예측 변수로, 재정 지원이 중퇴 예방의 핵심임을 확인했다.',
    'Enrolled 클래스를 제외하고 이진 분류로 재정의하는 것이 모델 성능과 서비스 실용성 측면에서 더 적합하였다.',
]:
    p = doc.add_paragraph(style='List Bullet')
    run = p.add_run(text)
    set_font(run)

add_heading(doc, '아쉬운 점 / 향후 과제', level=2)
add_table(doc,
    ['항목', '내용'],
    [
        ['데이터 일반화', '포르투갈 데이터 기반으로 국내 대학에 바로 적용하기 어려움. 국내 데이터로 재학습 필요'],
        ['Enrolled 예측 불가', '재학중 학생의 미래 중퇴 여부 예측 불가. 3클래스 모델 또는 시계열 접근 검토 필요'],
        ['LLM 연동', '학생이 자연어로 상황을 설명하면 LLM이 정보를 추출해 모델에 전달하는 방식으로 확장 계획'],
        ['모델 다양화', 'XGBoost, Logistic Regression 등 다른 알고리즘과의 성능 비교 및 앙상블 적용 검토'],
        ['설명 가능한 AI', 'SHAP 값을 활용해 개별 학생 예측 근거를 더 상세히 제공하는 방향으로 개선 가능'],
    ],
    col_widths=[4, 10])

# ── 저장 ──────────────────────────────────────────────────────
out = Path(__file__).parent / "보고서.docx"
doc.save(out)
print(f"저장 완료: {out}")
