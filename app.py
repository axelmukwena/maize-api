import os
import time
import pickle
import uvicorn
import inference
from tensorflow import keras
from typing import List
from fastapi import File, FastAPI

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

app = FastAPI()

# Load model and classes
MODEL = keras.models.load_model("model/", compile=False)
with open("model/lb.pickle", 'rb') as f:
    LB = pickle.load(f)


@app.get("/")
async def read_root():
    return {"message": "Welcome from the API"}


@app.post("/classify")
async def get_image(files: List[bytes] = File()):
    try:
        t = time.time()
        values, labels = inference.inference(MODEL, LB, files)
        print(time.time() - t)
        response = {
            "success": True,
            "values": values,
            "labels": labels,
            "message": "Predicted successfully."
        }
    except Exception:
        response = {"success": False, "message": "Server error! Contact support."}
    
    return response


if __name__ == "__main__":
    uvicorn.run(app)
