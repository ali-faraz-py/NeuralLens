import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
from io import BytesIO



model = ResNet50(weights='imagenet')

def predict_image(image_bytes):
    img = Image.open(BytesIO(image_bytes))
    img = img.resize((224, 224))
    
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    predictions = model.predict(img_array)
    decoded = decode_predictions(predictions, top=5)[0]

    return decoded