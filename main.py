import os
import time
import pickle
import sklearn
import uvicorn
import inference
from tensorflow import keras
from typing import List
from fastapi import UploadFile, FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
import json

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load model and classes
MODEL = keras.models.load_model("model/", compile=False)
with open("lb.pickle", 'rb') as f:
    LB = pickle.load(f)


@app.get("/")
async def read_root():
    return {"message": "Welcome from the API"}


@app.post("/classify")
async def get_image(files: List[UploadFile]):
    # try:
    # image = np.array(Image.open(ff))
    # print("image:", image)
    #
    t = time.time()
    predictions = inference.inference(MODEL, LB, files)
    # values, labels = [1, 2, 3, 4], [21, 234, 421, 312]
    print(time.time() - t)
    response = {
        "success": True,
        "predictions": predictions,
        "message": "Predicted successfully."
    }
    
    json_data = json.dumps(response)
    return Response(content=json_data, media_type="application/json")


if __name__ == "__main__":
    uvicorn.run(app)
