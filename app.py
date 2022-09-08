import json
import os
import time
from itertools import product

import biosppy
import numpy as np
from flask import Flask, request, send_file
from scipy.signal import filtfilt
from tensorflow.keras.models import load_model
import base64

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

app = Flask(__name__)
VERIFY_MODEL = load_model("model/", compile=False)
AUTHENTICATE_MODELS = {}
TEMPLATES = []
CLASSES = []
TEMPS = 0
TRIM = 20


def handle_classify(samples):
    pairs = [[t, s] for t, s in product(TEMPLATES, samples)]
    pairs = normalize(pairs)
    t = time.time()
    ps = VERIFY_MODEL.predict([pairs[:, 0], pairs[:, 1]])  # output looks like [[[1]], [[2]], [[3]]]
    ps = [i[0][0] for i in ps]
    print(time.time() - t)
    return ps


@app.route("/classify", methods=["POST"])
def classify():
    invalid = {"error": "Invalid request."}
    # ensure the file was properly uploaded to our endpoint
    if request.method == "POST":
        if 'file' in request.files:
            file = request.files['file']
            classes = ["Maize Streak Virus", "Healthy", "Fall Army Worm", "Unknown"]
            response = {
                # https://stackoverflow.com/a/70167203/8050183
                "success": True,
                "file": base64.b64encode(file).decode(),
                "label": random.choice(classes),
                "percent": 76,
            }
            return json.dumps(response)
        else:
            return json.dumps({"error": "Data sent is not valid."})
    return json.dumps(invalid)


@app.route("/")
def index():
    return json.dumps({"success": True})


if __name__ == '__main__':
    app.run()
