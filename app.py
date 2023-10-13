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
    # Loading trained model architecture and weights from saved file
    with open(architecture, 'r') as json_file:
        loaded_model_json = json_file.read()

    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(weights)

    # Open Class labels dictionary
    with open(classes_file, 'r') as class_file:
        classes = eval(class_file.read())

    print(request.form['imageBase64'])
    image_data = re.sub('^data:image/.+;base64,', '', request.form['imageBase64'])

    img = Image.open(BytesIO(base64.b64decode(image_data)))
    img.save("input" + str(randrange(1000)) + "__" + str(img_size) + "x" + str(img_size) + ".png")

    img = np.array(img)
    img = cv2.bitwise_not(img)

    # Convert to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)

    # Formatting and normalizing the image for the neural network
    img = img.reshape((1, img_size, img_size, 1))
    img = img.astype('float32') / 255

    # Running prediction on test image
    preds = loaded_model.predict(img)

    pred_result = dict(zip(range(len(preds[0])), preds[0]))
    sorted_result = sorted(pred_result.items(), key=operator.itemgetter(1), reverse=True)

    results = []
    for i in range(min(3, len(sorted_result))):
        identified_class = classes[sorted_result[i][0]]
        probability = str(sorted_result[i][1])
        results.append({'class': identified_class, 'probability': probability})

    print(*results, sep="\n")

    return results


if __name__ == '__main__':
    app.run()
