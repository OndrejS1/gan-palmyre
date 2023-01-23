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

    prediction = predict_with_weight_and_architecture(architecture, weights, 28)

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

    result = predict_with_weight_and_architecture(architecture, weights, 100)

    print(result)

    return make_response(result, 200)


def predict_with_weight_and_architecture(architecture, weights, img_size):
    classes = "static/classes_all.txt"
    # loading trained model architecture and weights from saved file
    json_file = open(architecture, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(weights)

    # file = flask.request.files['inputFile']
    # img = Image.open(file.stream)
    # Open Class labels dictionary. (human readable label given ID, alphabetically ordered, just as the classes were trained)
    classes = eval(open(classes, 'r').read())
    print(request.form['imageBase64'])
    image_data = re.sub('^data:image/.+;base64,', '', request.form['imageBase64'])
    img = Image.open(BytesIO(base64.b64decode(image_data)))
    img.save("input" + str(randrange(1000)) + "__" + str(img_size) + "x" + str(img_size) + ".png")
    # img = img.resize((img_size, img_size), Image.BILINEAR)
    img = np.array(img)
    img = cv2.bitwise_not(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

    # changing format to neural network readable and normalisation
    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
    img = img.astype('float32') / 255

    # running prediction on test image
    preds = loaded_model.predict(img)

    pred_result = dict(zip(range(len(preds[0])), preds[0]))

    sorted_result = sorted(pred_result.items(), key=operator.itemgetter(1), reverse=True)

    # print(preds)
    # print(np.argmax(preds[0], axis=0))

    identified_class_1 = classes[sorted_result[0][0]]
    probability_1 = str(sorted_result[0][1])

    identified_class_2 = classes[sorted_result[1][0]]
    probability_2 = str(sorted_result[1][1])

    identified_class_3 = classes[sorted_result[2][0]]
    probability_3 = str(sorted_result[2][1])

    return [
        {'class': identified_class_1, 'probability': probability_1},
        {'class': identified_class_2, 'probability': probability_2},
        {'class': identified_class_3, 'probability': probability_3}
    ]


if __name__ == '__main__':
    app.run()
