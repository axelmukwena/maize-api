import cv2
import numpy as np
from PIL import Image


def resize(img):
    size = 380
    h, w = img.shape[:2]
    min_size = np.amin([h, w])
    
    # Centralize and crop
    crop_img = img[int(h / 2 - min_size / 2):int(h / 2 + min_size / 2),
               int(w / 2 - min_size / 2):int(w / 2 + min_size / 2)]
    resized = cv2.resize(crop_img, (size, size), interpolation=cv2.INTER_AREA)
    return resized


def inference(model, lb, files):
    images = []
    for file in files:
        image = np.array(Image.open(file.file))
        resized = resize(image)
        images.append(resized)

    x = np.array(images)
    x = (x - x.min()) / (x.max() - x.min())

    predictions = model.predict(x)
    print("predictions:", predictions)
    print("lb.classes_:", lb.classes_)
    
    preds = []
    for i in range(x.shape[0]):
        pred_idx = np.argmax(predictions[i])
        prob = max(predictions[i])
        label = lb.classes_[pred_idx]
        preds.append([label, int(prob * 100)])
   
    return preds
