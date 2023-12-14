import operator
from io import BytesIO
from random import randrange

import base64
from flask import request, make_response
import re

import cv2
import flask
import numpy as np
import tensorflow as tf
from IPython.display import Image
from PIL import Image
from tensorflow.python.keras.models import model_from_json

from PredictionResponse import PredictionResponse
from flask_cors import CORS, cross_origin

app = flask.Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


def make_predict_response(classifier, probability):
    response = PredictionResponse(classifier, probability)
    return response


# predict by using handwritten model
@app.route("/predict-handwritten", methods=['POST'])
@cross_origin()
def predict_handwritten():
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True

    # used for passing arguments to script
    architecture = "static/handwritten_model_v3_color.json"
    weights = "static/handwritten_model_v3_color.h5"
    classes = "static/handwritten_classes.txt"

    prediction = predict_with_weight_and_architecture(architecture, weights, classes, 28)

    return prediction


# define a predict function as an endpoint
@app.route("/predict", methods=['POST'])
@cross_origin()
def predict():
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True

    # used for passing arguments to script
    architecture = "static/photo_gan2.json"
    weights = "static/photo_gan2.h5"
    classes = "static/photo_classes.txt"

    result = predict_with_weight_and_architecture(architecture, weights, classes, 100)

    print(result)

    return make_response(result, 200)


def predict_with_weight_and_architecture(architecture, weights, classes_file, img_size):
    # Load trained model architecture and weights
    with open(architecture, 'r') as json_file:
        loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(weights)

    # Open Class labels dictionary
    with open(classes_file, 'r') as class_file:
        classes = eval(class_file.read())

    # Extract and preprocess the image
    image_data = re.sub('^data:image/.+;base64,', '', request.form['imageBase64'])
    img = Image.open(BytesIO(base64.b64decode(image_data)))

    # Resize the image to the expected input size
    img = img.resize((img_size, img_size), Image.BILINEAR)
    img = np.array(img)

    # Ensure the image has 3 channels (BGR format)
    if img.shape[-1] == 4:  # Convert RGBA to BGR
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    elif img.shape[-1] == 1:  # Convert grayscale to BGR if needed
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Normalize the image
    img = img.astype('float32') / 255

    # Add a batch dimension to the image
    img = np.expand_dims(img, axis=0)

    # Run prediction on the processed image
    preds = loaded_model.predict(img)

    # Process prediction results
    pred_result = dict(zip(range(len(preds[0])), preds[0]))
    sorted_result = sorted(pred_result.items(), key=operator.itemgetter(1), reverse=True)

    results = []
    for i in range(min(3, len(sorted_result))):
        identified_class = classes[sorted_result[i][0]]
        probability = str(sorted_result[i][1])
        results.append({'class': identified_class, 'probability': probability})

    # Print the top 3 predictions
    print(*results, sep="\n")

    return results

if __name__ == '__main__':
    app.run()
