from flask import Flask, render_template, request, jsonify
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
from PIL import Image, ImageChops, ImageEnhance
import io

# Load your trained model
model = load_model('./model_02.h5')

# Define the image size expected by the model
image_size = (128, 128)

def convert_to_ela_image(path, quality):
    temp_filename = 'temp_file_name.jpg'
    ela_filename = 'temp_ela.png'
    
    image = Image.open(path).convert('RGB')
    image.save(temp_filename, 'JPEG', quality = quality)
    temp_image = Image.open(temp_filename)
    
    ela_image = ImageChops.difference(image, temp_image)
    
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff
    
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
    
    return ela_image

def prepare_image(image_path):
    return np.array(convert_to_ela_image(image_path, 90).resize(image_size)).flatten() / 255.0


app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])  # Handle both GET and POST requests
def upload_file():
    if request.method == 'POST':
        image = request.files['image']
        image = prepare_image(image)
        image = image.reshape(-1, 128, 128, 3)
        prediction = None
        try:
            prediction = model.predict(image)   
            class_index = np.argmax(prediction, axis=-1)[0]
            class_names = ['Fake', 'Real']
            prediction = class_names[class_index]

            if prediction is not None:
                return jsonify({'prediction': prediction})
            else:
                return jsonify({'error': 'No prediction made'})
        except Exception as e:
            return jsonify({'error': str(e)})  # Handle prediction errors

    else:
        return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
