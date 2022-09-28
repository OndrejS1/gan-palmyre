import dataclasses
import json

import cv2
import flask
import numpy as np
import tensorflow as tf
from IPython.display import Image
from PIL import Image
from tensorflow.python.keras.models import model_from_json

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

    # used for passing arguments to script
    architecture = "static/handwritten_model_v3_color.json"
    weights = "static/handwritten_model_v3_color.h5"

    return predict_with_weight_and_architecture(architecture, weights, 100)


# predict by using handwritten model
@app.route("/predict-handwritten", methods=['POST'])
def predict():
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True

    # used for passing arguments to script
    architecture = "static/photo_gan2.json"
    weights = "static/photo_gan2.h5"

    return predict_with_weight_and_architecture(architecture, weights, 28)


def predict_with_weight_and_architecture(architecture, weights, img_size):
    classes = "static/classes_all.txt"
    # loading trained model architecture and weights from saved file
    json_file = open(architecture, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(weights)

    # Open Class labels dictionary. (human readable label given ID, alphabetically ordered, just as the classes were trained)
    classes = eval(open(classes, 'r').read())
    file = flask.request.files['inputFile']
    img = Image.open(file.stream)
    img = img.resize((img_size, img_size), Image.BILINEAR)
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    img = np.array(img)

    # changing format to neural network readable and normalisation
    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
    img = img.astype('float32') / 255

    # running prediction on test image
    preds = loaded_model.predict(img)

    response = app.response_class(
        response=json.dumps(make_predict_response(classes[np.argmax(preds)], str(preds[0][np.argmax(preds)])),
                            cls=EnhancedJSONEncoder))
    return response


class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return super().default(o)


if __name__ == '__main__':
    app.run()
