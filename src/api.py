from fastapi import FastAPI, HTTPException
import joblib
from sklearn.metrics.pairwise import cosine_similarity

data=joblib.load("data/processed/tfidf_vectors.pkl")


project_vectors=data["project_vectors"]
user_vectors=data["user_vectors"]
projects=data["projects"]
users=data["users"]

app=FastAPI(
    title="Project Recommendation API",
    description="AI-powered project recommender using TF-IDF and cosine similarity",
    version="1.0"
)

@app.get("/")
def health_check():
    return {"status": "API is running"}

@app.get("/recommend/{user_id}")

def recommend_projects(user_id: int, top_n: int = 5):
    if user_id not in users["user_id"].values:
        raise HTTPException(status_code=404, detail="User not found")
    
    user_idx=users.index[users["user_id"]==user_id][0]

    similarity_scores=cosine_similarity(user_vectors[user_idx],project_vectors)[0]

    project_with_scores=projects.copy()

    project_with_scores["similarity_score"]=similarity_scores

    top_projects=project_with_scores.sort_values(by="similarity_score",ascending=False).head(top_n)

    return top_projects[["project_id", "title", "difficulty", "tags", "similarity_score"]].to_dict(orient="records")
