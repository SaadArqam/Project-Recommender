import pandas as pd
import numpy as np
from pathlib import Path

raw_data_path=Path("data/raw/users.csv")
processed_data_path=Path("data/processed/users_clean.csv")

def clean_users_data():
    df=pd.read_csv(raw_data_path)
    df.columns = [col.lower().strip() for col in df.columns]

    txt_cols=["skills","experience_level","interests"]

    for col in txt_cols:
        df[col]=(df[col].astype(str).str.lower().str.strip())

    df["skills"]=df["skills"].str.replace(r"\s*,\s*", ",", regex=True)
    df["interests"]=df["interests"].str.replace(r"\s*,\s*", ",", regex=True)

    df.drop_duplicates(subset="user_id",inplace=True)

    processed_data_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_data_path,index=False)

    print("Cleaned dataset saved to",processed_data_path)

if __name__=="__main__":
    clean_users_data()




