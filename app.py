import operator
from io import BytesIO
from random import randrange

import base64
from flask import request, make_response, json
import re
import os
from ultralytics import YOLO
import shutil

import cv2
import flask
import numpy as np
import tensorflow as tf
from IPython.display import Image
from PIL import Image
from matplotlib import pyplot as plt
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

    print(request.form['imageBase64'])
    img = extract_image_from_request()
    img.save("input" + str(randrange(1000)) + "__" + str(img_size) + "x" + str(img_size) + ".png")

    # Resize the image to the expected input size
    img = img.resize((img_size, img_size), Image.BILINEAR)
    img = np.array(img)

    if architecture == "static/photo_gan2.json":
        if img.shape[-1] == 4:  # Convert RGBA to BGR
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        elif img.shape[-1] == 1:  # Convert grayscale to BGR if needed
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        if img.shape[-1] == 3:  # Convert RGB to Grayscale
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif img.shape[-1] == 4:  # Convert RGBA to Grayscale
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
        # Add the necessary channel dimension (1 for grayscale)
        img = np.expand_dims(img, axis=-1)

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


def extract_image_from_request():
    image_data = re.sub('^data:image/.+;base64,', '', request.form['imageBase64'])
    img = Image.open(BytesIO(base64.b64decode(image_data)))
    return img


def read_class_names_from_file(filename):
    class_names = {}
    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split(':')
            class_index, class_name = int(parts[0]), parts[1].strip().strip('"')
            class_names[class_index] = class_name
    return class_names


def extract_polygons_from_file(filename):
    with open(filename, 'r') as file:
        return [
            {'class_index': int(parts[0]), 'coordinates': [float(coord) for coord in parts[1:]]}
            for parts in [line.strip().split(' ') for line in file]
        ]


def extract_polygons_directly_from_results2(results):
    polygons = []
    for detection in results:
        # Assuming each detection has the format [x1, y1, x2, y2, confidence, class_index]
        x1, y1, x2, y2, _, class_index = detection
        # Convert bounding box to your polygon format
        coordinates = [x1, y1, x2, y1, x2, y2, x1, y2]
        polygons.append({'class_index': int(class_index), 'coordinates': coordinates})
    return polygons


def calculate_avg_size(polygons):
    heights = [max(p['coordinates'][1::2]) - min(p['coordinates'][1::2]) for p in polygons]
    widths = [max(p['coordinates'][::2]) - min(p['coordinates'][::2]) for p in polygons]
    return np.average(heights), np.average(widths)


def get_polygon_center(polygon):
    x_coords, y_coords = polygon['coordinates'][::2], polygon['coordinates'][1::2]
    return sum(x_coords) / len(x_coords), sum(y_coords) / len(y_coords)


def sort_polygons_in_rows(polygons):
    avg_height, _ = calculate_avg_size(polygons)
    threshold_height = avg_height / 2

    sorted_polygons = sorted(polygons, key=lambda p: get_polygon_center(p)[1])
    rows, current_row = [], []

    for polygon in sorted_polygons:
        if current_row and abs(
                get_polygon_center(polygon)[1] - get_polygon_center(current_row[-1])[1]) > threshold_height:
            rows.append(sorted(current_row, key=lambda p: get_polygon_center(p)[0], reverse=True))
            current_row = []
        current_row.append(polygon)

    if current_row:
        rows.append(sorted(current_row, key=lambda p: get_polygon_center(p)[0], reverse=True))
    return rows


def visualize_polygons(polygons, class_names):
    colormap = plt.cm.get_cmap('tab10', len(set(p['class_index'] for p in polygons)))
    plt.figure(figsize=(6, 12))

    for p in polygons:
        x_coords, y_coords = p['coordinates'][::2], p['coordinates'][1::2]
        class_color = colormap(p['class_index'] % colormap.N)
        plt.fill(x_coords + [x_coords[0]], y_coords + [y_coords[0]], color=class_color)

        label_x, label_y = get_polygon_center(p)
        plt.text(label_x, label_y + 0.02, class_names[p['class_index']], color='black', fontsize=8, ha='center')

    plt.xlabel('X-coordinate')
    plt.ylabel('Y-coordinate')
    plt.title('Sorted Polygons')
    plt.gca().invert_yaxis()

    dirPath = 'static/augmented'
    if not os.path.isdir(dirPath):
        os.makedirs(dirPath)

    plt.savefig(dirPath + '/Fig1.png', bbox_inches='tight')


def remove_file_with_pattern():
    directory_to_remove = 'runs/segment'

    # Remove the directory
    if os.path.exists(directory_to_remove):
        shutil.rmtree(directory_to_remove)


def make_convert_augmented_response(response, image):
    # Encode the placeholder image data to base64
    # Create a JSON object
    json_data = {
        "transcript": response,
        "image": image
    }

    # Convert the JSON object to a string
    json_string = json.dumps(json_data, indent=4)

    return json_string


def load_and_remove_image(image_path):
    """
    Load a PNG image from the file system, convert it to base64 for browser display,
    and then remove the image from the file system.

    :param image_path: The path to the image file.
    :return: Base64 encoded string of the image.
    """
    try:
        # Check if the file exists and is a PNG image
        if os.path.exists(image_path) and image_path.lower().endswith('.png'):
            # Read and encode the image to base64
            with open(image_path, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

            # Remove the image file from the file system
            os.remove(image_path)

            return 'data:image/png;base64,' + encoded_image
        else:
            return "File not found or is not a PNG image."
    except Exception as e:
        return str(e)


@app.route("/convert-augmented", methods=['POST'])
@cross_origin()
def convert_augmented():
    class_names = read_class_names_from_file("static/class_list_polygons.txt")
    model = YOLO('static/multiclass_v21_best.pt')
    img = extract_image_from_request()

    # img is already a PIL image, no need to open it again
    opencv_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    file_path = 'static/output_image.png'
    # Save using OpenCV
    cv2.imwrite(file_path, opencv_image)
    results = model.predict(source=file_path, save=True, save_txt=True)
    polygons = extract_polygons_from_file('runs/segment/predict/labels/output_image.txt')

    load_and_remove_image(file_path)
    remove_file_with_pattern()

    sorted_polygons_rows = sort_polygons_in_rows(polygons)

    response = [[]]

    for i, row in enumerate(sorted_polygons_rows):
        response.append([class_names[p['class_index']] for p in row])

    sorted_polygons = [p for row in sorted_polygons_rows for p in row]
    visualize_polygons(sorted_polygons, class_names)

    augmented_image = load_and_remove_image('static/augmented/Fig1.png')
    return make_convert_augmented_response(response, augmented_image)


if __name__ == '__main__':
    app.run()
