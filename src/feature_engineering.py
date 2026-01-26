import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

USERS_PATH = Path("data/processed/users_clean.csv")
PROJECTS_PATH = Path("data/processed/projects_clean.csv")
MODEL_DIR = Path("data/processed/models")

def build_tfidf_vectors():
    users = pd.read_csv(USERS_PATH)
    projects = pd.read_csv(PROJECTS_PATH)

    users["combined_text"] = users["skills"] + " " + users["interests"]
    projects["combined_text"] = (
        projects["required_skills"] + " " +
        projects["tags"] + " " +
        projects["description"]
    )

    vectorizer = TfidfVectorizer(stop_words="english")
    project_vectors = vectorizer.fit_transform(projects["combined_text"])
    user_vectors = vectorizer.transform(users["combined_text"])

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(vectorizer, MODEL_DIR / "tfidf_vectorizer.pkl")
    joblib.dump(project_vectors, MODEL_DIR / "project_vectors.pkl")
    joblib.dump(user_vectors, MODEL_DIR / "user_vectors.pkl")

    print("✅ TF-IDF artifacts saved successfully")

if __name__ == "__main__":
    build_tfidf_vectors()
