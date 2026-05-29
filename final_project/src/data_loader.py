import pandas as pd
import streamlit as st
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / "data" / "dataset.csv"

@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    return df

def get_feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c != "Target"]
