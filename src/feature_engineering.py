import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# Paths
USERS_PATH = Path("data/processed/users_clean.csv")
PROJECTS_PATH = Path("data/processed/projects_clean.csv")
VECTORS_PATH = Path("data/processed/tfidf_vectors.pkl")

def build_tfidf_vectors():
    # Load cleaned data
    users = pd.read_csv(USERS_PATH)
    projects = pd.read_csv(PROJECTS_PATH)

    # Combine user text features
    users["combined_text"] = (
        users["skills"] + " " + users["interests"]
    )

    # Combine project text features
    projects["combined_text"] = (
        projects["required_skills"] + " " +
        projects["tags"] + " " +
        projects["description"]
    )

    # Fit TF-IDF on projects (content-based)
    vectorizer = TfidfVectorizer(stop_words="english")

    project_vectors = vectorizer.fit_transform(
        projects["combined_text"]
    )

    user_vectors = vectorizer.transform(
        users["combined_text"]
    )

    # Save everything
    VECTORS_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "vectorizer": vectorizer,
            "project_vectors": project_vectors,
            "user_vectors": user_vectors,
            "projects": projects,
            "users": users
        },
        VECTORS_PATH
    )

    print("✅ TF-IDF vectors saved to:", VECTORS_PATH)

if __name__ == "__main__":
    build_tfidf_vectors()
