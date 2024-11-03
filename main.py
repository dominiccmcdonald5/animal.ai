from fastapi import FastAPI
from AI.animalName import *
from pydantic import BaseModel

app = FastAPI()

class ClassificationRequest(BaseModel):
    image_path: str


@app.get("/")
async def root():
    return {"message": "Hello World!!!"}


@app.post("/predict/")
async def predict(request_body: ClassificationRequest):
    label_encoder, model = training()
    prediction = predict_image(request_body.image_path, label_encoder, model)

    return {"message": prediction}


# @app.post("/test/")
# async def test(request_body: ClassificationRequest):
#     return {"message": f"{request_body}"}
