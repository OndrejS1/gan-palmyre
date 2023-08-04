import base64
import requests

# Open the image file in binary mode, encode it in base64, and decode it to ASCII
#C:\\Users\\Ondrej.Svojse\\Downloads\\IMG_2425.jpg
with open("path_to_your_image", "rb") as image_file:
    encoded_image = base64.b64encode(image_file.read()).decode('ascii')

# Send a POST request to the /predict-handwritten endpoint with the encoded image in the form data
response = requests.post("http://localhost:5000/predict-handwritten", data={"imageBase64": encoded_image})

# Print the response from the server
print(response.json())