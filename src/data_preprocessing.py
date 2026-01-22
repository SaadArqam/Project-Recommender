import pandas as pd
import numpy as np
from pathlib import Path

raw_data_path=Path("data/raw/projects.csv")
processed_data_path=Path("data/processed/projects_clean.csv")

def clean_projects_data():
    df=pd.read_csv(raw_data_path)
    df.columns = [col.lower().strip() for col in df.columns]

    txt_cols=["title","description","required_skills","tags"]
    for col in txt_cols:
        df[col]=(df[col].astype(str).str.lower().str.strip())
        
    