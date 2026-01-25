# Intelligent Project Recommendation System (AI/ML)

## Overview
This project is a content-based recommendation system that suggests relevant projects to users based on their skills, interests, and experience level.  
It uses data mining techniques, TF-IDF vectorization, and cosine similarity to generate personalized recommendations and exposes the model via a REST API using FastAPI.

---

## Problem Statement
Finding suitable projects for developers based on their skill set and interests is challenging.  
This system aims to automatically match users with relevant projects by analyzing textual features and learning similarity patterns.

---

## Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- FastAPI
- TF-IDF Vectorization
- Cosine Similarity

---

## Dataset
- Custom-created datasets simulating real-world users and projects
- Data initially created and validated using Google Sheets
- Exported as CSV and processed using Python

### Users Data
- skills (comma-separated)
- experience_level (Beginner / Intermediate / Advanced)
- interests (comma-separated)

### Projects Data
- required_skills
- tags
- difficulty
- description

---


## Architecture

- **Raw CSV Data**
- **Data Preprocessing** (cleaning & normalization)
- **Feature Engineering** (TF-IDF vectorization)
- **Cosine Similarity Computation**
- **Top-N Project Recommendations**
- **FastAPI REST API**



---

## How It Works
1. Textual features from users and projects are combined
2. TF-IDF converts text into numerical vectors
3. Cosine similarity measures relevance between users and projects
4. Projects are ranked and top-N recommendations are returned

---

## API Endpoints

### GET /

- Get Recommendations


### GET /recommend/{user_id}?top_n=5


- Returns a ranked list of recommended projects with similarity scores.

---

## How to Run Locally

```bash
# create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# install dependencies
pip install -r requirements.txt

# run feature engineering
python src/feature_engineering.py

# start API
uvicorn src.api:app --reload

Open:
http://127.0.0.1:8000/docs

```



## Key Learnings

- Data preprocessing and feature engineering for ML pipelines
- Content-based recommendation systems
- Text vectorization using TF-IDF
- Deploying ML models using FastAPI
- Debugging and API integration

---

## Future Improvements

- Difficulty-aware recommendations
- Hybrid recommender (content + popularity)
- Frontend integration
- Model evaluation metrics


