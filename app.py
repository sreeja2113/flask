from flask import Flask, request
from keras.models import load_model
from PIL import Image
import numpy as np
from flask_cors import CORS
from pymongo import MongoClient
from bson.binary import Binary

app = Flask(__name__)
CORS(app, origins=["http://localhost:3000"])

client = MongoClient('mongodb+srv://sreeja:sreejayayy@cluster0.tth9dwr.mongodb.net/?retryWrites=true&w=majority')
db = client['test']
collection = db['cellimages']

model = load_model("E:/ps2/malaria_cnn_model.h5")

def preprocess_image(image):
    image = image.resize((50, 50))
    image_array = np.array(image) 
    image_array=image_array.astype('float32')/ 255.0
    image_array= np.expand_dims(image_array, axis=0)
    return image_array

@app.route("/predict", methods=["POST"])
def predict():
    imagefile = request.files["imagefile"]
    image = Image.open(imagefile)
    image_array = preprocess_image(image)
    pred = model.predict(image_array)
    #prediction = np.argmax(pred)
    prediction = "Parasitized" if pred[0][0] > 0.5 else "Uninfected"
    # save the image to the MongoDB database
    image_binary = Binary(imagefile.read())
    collection.insert_one({'image_file': image_binary,'prediction': prediction})
    return str(pred[0])
    #return prediction

if __name__ == "_main_":
    app.run(debug=True)
