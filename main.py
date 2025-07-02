import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# FastAPI 앱 생성
app = FastAPI()

# 입력 데이터 모델 정의
class FeatureInput(BaseModel):
    view: float
    rating: float
    review: float
    genre: float

# 추천 점수 계산 함수
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

@app.post("/recommend")
def recommend(input: FeatureInput):
    features = [input.view, input.rating, input.review, input.genre]
    weights = [0.4, 0.3, 0.2, 0.1]
    score = sum([f * w for f, w in zip(features, weights)])
    prob = sigmoid(score)
    return {"score": round(score, 4), "prob": round(prob, 4)}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
