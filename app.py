import gradio as gr
import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path



MODEL_DIR = Path("data/processed/models")

projects = pd.read_csv("data/processed/projects_clean.csv")
users = pd.read_csv("data/processed/users_clean.csv")



vectorizer = joblib.load(MODEL_DIR / "tfidf_vectorizer.pkl")
project_vectors = joblib.load(MODEL_DIR / "project_vectors.pkl")


def extract_unique_values(column):
    values = set()
    for row in users[column].dropna():
        for item in row.split(","):
            values.add(item.strip())
    return sorted(values)

ALL_SKILLS = extract_unique_values("skills")
ALL_INTERESTS = extract_unique_values("interests")
ALL_DIFFICULTIES = sorted(projects["difficulty"].unique())


def recommend(skills, interests, difficulty, top_n):
    if not skills and not interests:
        return "Please select at least one skill or interest."


    query_text = " ".join(skills + interests)


    query_vector = vectorizer.transform([query_text])


    scores = cosine_similarity(query_vector, project_vectors)[0]


    results = projects.copy()
    results["score"] = scores


    if difficulty != "Any":
        results = results[results["difficulty"] == difficulty]


    top_projects = results.sort_values(
        by="score", ascending=False
    ).head(top_n)

    return top_projects[
        ["title", "difficulty", "tags", "score"]
    ]


demo = gr.Interface(
    fn=recommend,
    inputs=[
        gr.CheckboxGroup(ALL_SKILLS, label="Select Skills"),
        gr.CheckboxGroup(ALL_INTERESTS, label="Select Interests"),
        gr.Dropdown(
            ["Any"] + ALL_DIFFICULTIES,
            value="Any",
            label="Difficulty Level"
        ),
        gr.Slider(1, 10, value=5, step=1, label="Top N Projects")
    ],
    outputs=gr.Dataframe(
        headers=["Title", "Difficulty", "Tags", "Score"]
    ),
    title="AI Project Recommendation System",
    description="Select your skills, interests, and difficulty to get personalized project recommendations."
)

if __name__ == "__main__":
    demo.launch()
