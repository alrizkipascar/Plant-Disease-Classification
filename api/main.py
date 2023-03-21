from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf


app = FastAPI()

MODEL = tf.keras.models.load_model("C:/Users/Alrziki/Documents/Python/Grinding Machine Learning/Potatoe-Disease-Classification/models/1")

origins = [
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/ping")

async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    #img_batch = np.expand_dims(image,0)
    #prediction = MODEL.predict(img_batch)
    return image

@app.post("/predict")

async def predict(
    file: UploadFile
):
    image = read_file_as_image(await file.read())
    return image
    

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)