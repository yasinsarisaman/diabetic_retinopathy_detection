# load_model_sample.py
from keras.models import load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import os


def load_image(img_path, show=False):

    img = image.load_img(img_path, target_size=(64, 64))
    img_tensor = image.img_to_array(img)                   
    img_tensor = np.expand_dims(img_tensor, axis=0)        
    img_tensor /= 255.                                      

    if show:
        plt.imshow(img_tensor[0])                           
        plt.axis('off')
        plt.show()

    return img_tensor

# load model
model = load_model("model.h5")

def predict_single_img(img_path):
    print(img_path)
    new_image = load_image(img_path) # load a single image
    pred_result = model.predict(new_image) # check prediction
    return pred_result
    

def send_prediction_result(pred_res):
    if pred_res<0.15:
        prediction = "NO DIABETIC RETINOPATHY"
    else:
        prediction = "DIABETIC RETINOPATHY"
    return prediction;