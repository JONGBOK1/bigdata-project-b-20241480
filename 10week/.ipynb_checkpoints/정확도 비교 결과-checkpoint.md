##기존 프롬프트

PROMPT_TEMPLATE = 'You are a web security expert. Classify each HTTP request as "Normal" or "Anomalous" and provide a brief reason.\n\nExamples:\nRequest: GET /index.jsp HTTP/1.1\nOutput: {{"label": "Normal", "reason": "Standard page request, no suspicious pattern"}}\n\nRequest: GET /search?q=\' OR \'1\'=\'1 HTTP/1.1\nOutput: {{"label": "Anomalous", "reason": "Classic SQL Injection pattern with OR 1=1"}}\n\nNow classify:\nRequest: {http_text}\nOutput:'

#정확도

LLM 정확도: 0.8500
LLM F1:    0.8624
분류 실패(Unknown): 1건

              precision    recall  f1-score   support

      Normal       0.86      0.81      0.84        47
   Anomalous       0.84      0.89      0.86        53

    accuracy                           0.85       100
   macro avg       0.85      0.85      0.85       100
weighted avg       0.85      0.85      0.85       100

## 변경된 프롬프트

PROMPT_TEMPLATE = '''You are a Web Application Firewall (WAF) AI.
Analyze the HTTP request and strictly output a JSON object with two keys: "analysis" and "label".
"label" must be either "Normal" or "Anomalous".

Rules:
1. "Normal": Standard web traffic, API calls, standard parameters. Do not flag just because a URL is long.
2. "Anomalous": Contains SQL Injection (', OR 1=1, UNION), XSS (<script>, javascript:), or Path Traversal (../).

Examples:
Request: GET /index.jsp?user=test&page=2 HTTP/1.1
Output: {{"analysis": "Standard parameters with no malicious payload detected.", "label": "Normal"}}

Request: GET /search?q=' OR '1'='1 HTTP/1.1
Output: {{"analysis": "Found an apostrophe followed by a tautology (OR 1=1), which is a classic SQL Injection.", "label": "Anomalous"}}

Request: {http_text}
Output:'''

# 변경된 정확도

LLM 정확도: 0.5200
LLM F1:    0.6842
분류 실패(Unknown): 0건

              precision    recall  f1-score   support

      Normal       0.00      0.00      0.00        47
   Anomalous       0.53      0.98      0.68        53

    accuracy                           0.52       100
   macro avg       0.26      0.49      0.34       100
weighted avg       0.28      0.52      0.36       100


## 비교

정확도가 52%로 폭락한 원인은 첨부해주신 지표에 아주 명확하게 나와 있습니다. 모델이 정상(Normal) 데이터를 단 하나도 맞추지 못했습니다 (Normal의 Recall이 0.00). 즉, 100개의 데이터(정상 47개, 공격 53개) 중 모델이 "전부 다 공격(Anomalous)이다!" 라고 무지성으로 찍어버린 것입니다. (그래서 공격 53개 중 52개를 맞춰 Anomalous Recall은 0.98로 높게 나온 것입니다).