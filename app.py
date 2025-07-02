import streamlit as st
from PIL import Image
import numpy as np
import os
import pickle
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import matplotlib.pyplot as plt

# Load the tokenizer, model, image features, and caption mapping
tokenizer = pickle.load(open("./tokenizer_inception.pkl", 'rb'))
features = pickle.load(open("./features_inception.pkl", 'rb'))
model = load_model("./best_model_inception.keras")
mapping = pickle.load(open("./mapping_inception.pkl", 'rb'))  # Load caption mapping (add your actual path)

# Define max length (used during training)
max_length = 35  # Set this to the value you used in training

# Helper function to convert index to word
def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# Function to generate caption
def predict_caption(model, image, tokenizer, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], max_length)
        yhat = model.predict([image, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idx_to_word(yhat, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    final_caption = in_text.replace('startseq', '').replace('endseq', '').strip()
    return final_caption

# Streamlit GUI setup
st.title("Image Caption Generator")

# Upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Predict button
    if st.button("Generate Caption"):
        # Get the image ID from the filename
        image_id = uploaded_file.name.split('.')[0]

        if image_id in features:
            # Display predicted caption
            st.write('Generated Caption for above image...')
            y_pred = predict_caption(model, features[image_id], tokenizer, max_length)
            st.write(y_pred)
        else:
            st.write("Features for this image are not available.")