from transformers import pipeline
import torch

# GPU/CPU 자동 감지
device = 0 if torch.cuda.is_available() else -1

# ─── 1. 감성 분석 (한국어 지원) ───
print("=" * 60)
print("1. 다국어 감성 분석")
print("=" * 60)

classifier = pipeline(
    "sentiment-analysis",
    model="nlptown/bert-base-multilingual-uncased-sentiment",
    device=device
)