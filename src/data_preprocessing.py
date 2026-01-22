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

    df["required_skills"]=df["required_skills"].str.replace(r"\s*,\s*", ",", regex=True)
    df["tags"]=df["tags"].str.replace(r"\s*,\s*", ",", regex=True)
    df.drop_duplicates(subset="title",inplace=True)

    processed_data_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_data_path,index=False)

    print("Cleaned dataset and saved to",processed_data_path)

if __name__=="__main__":
    clean_projects_data()
    