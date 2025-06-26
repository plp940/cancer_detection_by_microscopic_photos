import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

IMAGE_SIZE = 256
MODEL_PATH = 'LungCancerPrediction.h5'
CLASS_NAMES = ['Benign', 'Malignant', 'Normal']

model = tf.keras.models.load_model(MODEL_PATH)

st.set_page_config(page_title="Lung Cancer Prediction", layout="centered")
st.title("Lung Cancer Identification")
st.markdown("Upload a lung cell image to predict the type of lung condition.")



#model = tf.keras.models.load_model(model_path)
#model = load_model('lung_cancer_model.keras')
#for uploading a image

img = st.file_uploader("Choose a image file", type=["jpg", "jpeg", "png","webp"])   
# Check if an image file is uploaded
if img is not None and img.name.endswith(('jpg', 'jpeg', 'png','webp')):
    # Display the image
    image = Image.open(img)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    # --- Image preprocessing steps ---
    image = image.resize((256, 256))  # replace with your model’s input size
    image_array = np.array(image)

    #🎯 Optional: Auto-detect input size
#You can dynamically get the expected input shape like this:
#input_size = model.input_shape[1:3]  # (height, width)
#Image = image.resize(input_size)


    if image_array.shape[-1] == 4:  # RGBA to RGB
        image_array = image_array[:, :, :3]

    image_array = image_array / 255.0  # normalize if model was trained on normalized images
    image_array = np.expand_dims(image_array, axis=0)  # add batch dimension
    class_name_map = {
    "lung_acc": "Adenocarcinoma (Cancerous)",
    "lung_n": "Normal (Non-Cancerous)",
    "lung_scc": "Squamous Carcinoma (Cancerous)"
     }
     # List must match order in which the model was trained
    original_class_names = ["lung_acc", "lung_n", "lung_scc"]

      # Make prediction
    prediction = model.predict(image_array)
    predicted_class = np.argmax(prediction)
    predicted_key = original_class_names[predicted_class]
    predicted_label = class_name_map[predicted_key]

# Show result
    st.success(f"Prediction: {predicted_label} (Confidence: {prediction[0][predicted_class]:.2f})")

    
