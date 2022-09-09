import pickle
from tensorflow import keras

MODEL = keras.models.load_model("model/", compile=False)

pkl_filename = 'model.pkl'
with open(pkl_filename, 'wb') as file:
    pickle.dump(MODEL, file)
