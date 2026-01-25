import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

data=joblib.load("data/processed/tfidf_vectors.pkl")

vectorizer=data["vectorizer"]
project_vectors=data["project_vectors"]
user_vectors=data["user_vectors"]
projects=data["projects"]
users=data["users"]


def recommend_projects_for_user(user_id,top_n=5):

    user_idx=users.index[users["user_id"]==user_id][0]

    similarity_score=cosine_similarity(user_vectors[user_idx],project_vectors)[0]

    project_with_scores=projects.copy()

    project_with_scores["similarity_score"]=similarity_score

    top_projects=project_with_scores.sort_values(by="similarity_score",ascending=False).head(top_n)

    return (top_projects[["project_id", "title", "difficulty", "tags", "similarity_score"]])


if __name__=="__main__":
    result=recommend_projects_for_user(user_id=1,top_n=5)
    print(result)



