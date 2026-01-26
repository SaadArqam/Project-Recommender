import gradio as gr
import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

MODEL_DIR = Path("data/processed/models")

vectorizer = joblib.load(MODEL_DIR / "tfidf_vectorizer.pkl")
project_vectors = joblib.load(MODEL_DIR / "project_vectors.pkl")
user_vectors = joblib.load(MODEL_DIR / "user_vectors.pkl")

projects = pd.read_csv("data/processed/projects_clean.csv")
users = pd.read_csv("data/processed/users_clean.csv")


def recommend(user_id, top_n):
    user_id = int(user_id)

    if user_id not in users["user_id"].values:
        return "User not found"

    user_idx = users.index[users["user_id"] == user_id][0]

    scores = cosine_similarity(
        user_vectors[user_idx], project_vectors
    )[0]

    projects_copy = projects.copy()
    projects_copy["score"] = scores

    top_projects = projects_copy.sort_values(
        by="score", ascending=False
    ).head(top_n)

    return top_projects[
        ["title", "difficulty", "tags", "score"]
    ]

demo = gr.Interface(
    fn=recommend,
    inputs=[
        gr.Number(label="User ID", precision=0),
        gr.Slider(1, 10, value=5, step=1, label="Top N Projects")
    ],
    outputs=gr.Dataframe(
        headers=["Title", "Difficulty", "Tags", "Score"]
    ),
    title="AI Project Recommendation System",
    description="Get personalized project recommendations based on skills and interests"
)

if __name__ == "__main__":
    demo.launch()
