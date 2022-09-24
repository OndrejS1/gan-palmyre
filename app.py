import json

from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json
import numpy as np
import cv2
import tensorflow as tf
import os
from IPython.display import Image, display
import argparse
from PIL import Image


import flask

from PredictionResponse import PredictionResponse

app = flask.Flask(__name__)


def make_predict_response(classifier, probability):
    response = PredictionResponse(classifier, probability)
    return response



# define a predict function as an endpoint
@app.route("/predict", methods=['POST'])
def predict():
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True

    #used for passing arguments to script
    architecture = "static/photo_gan2.json"
    weights = "static/photo_gan2.h5"
    classes = "static/classes_all.txt"
    img_path = "static/beth_0005.png"

    #loading trained model architecture and weights from saved file
    json_file = open(architecture, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(weights)

    # Open Class labels dictionary. (human readable label given ID, alphabetically ordered, just as the classes were trained)
    classes = eval(open(classes, 'r').read())

    #opening image for classification
    #img = Image.open(img_path)

    file = flask.request.files['inputFile']
    img = Image.open(file.stream)
    img = img.resize((100, 100), Image.BILINEAR)
    img = np.array(img)
    print(img.shape)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    img = np.array(img)
    print(img.shape)

    #changing format to neural network readable and normalisation
    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
    img = img.astype('float32') / 255

    #running prediction on test image
    preds = loaded_model.predict(img)

    print("Class is: " + classes[np.argmax(preds)])
    print("Certainty is: " + str(preds[0][np.argmax(preds)]))

    response = app.response_class(
        response=json.dumps(make_predict_response(classes[np.argmax(preds)], str(preds[0][np.argmax(preds)])), cls=EnhancedJSONEncoder,
    ))

    return response


import dataclasses, json

class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return super().default(o)

if __name__ == '__main__':
    app.run()
