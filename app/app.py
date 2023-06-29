import streamlit as st
import down
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image

import cv2
import numpy as np
import matplotlib.pyplot as plt

from visualization import visualize_image

import keras.backend.tensorflow_backend as tb
tb._SYMBOLIC_SCOPE.value = True

model_path = 'model/model.h5'
model = models.load_model(model_path, backbone_name='resnet50')

labels_to_names = {0: 'Bookcase', 1: 'Bathtub', 2: 'Pillow', 3: 'Couch', 4: 'Gas stove', 5: 'Washing machine', 6: 'Bed',
                  7: 'Refrigerator', 8: 'Bathroom accessory', 9: 'Kitchen & dining room table', 10: 'Television', 11: 'Sink',
                  12: 'Sofa bed', 13: 'Kitchenware', 14: 'Toilet', 15: 'Ceiling fan', 16: 'Microwave oven', 17: 'Furniture',
                  18: 'Coffeemaker', 19: 'Cupboard', 20: 'Dishwasher', 21: 'Shower', 22: 'Clock', 23: 'Countertop',
                  24: 'Mug', 25: 'Table'}

def show(image_path):
    image = read_image_bgr(image_path)

    # copy to draw on
    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

    # preprocess image for network
    image = preprocess_image(image)
    image, scale = resize_image(image)

    boxes, scores, labels = model.predict(np.expand_dims(image, axis=0))
    boxes /= scale

    st.write("Things found:")

    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        if score < 0.5:
            break

        visualize_image(draw, box, score, label)
        st.write(labels_to_names[label], 'with score:', score)

    plt.figure(figsize=(15, 15))
    plt.axis('off')
    plt.imshow(draw)
    st.pyplot()

st.title("Household Object Detection")
st.write("Upload an image here and let machine learning predict what object is in the image.")
uploaded_image = st.file_uploader("Choose a png or jpg image",
                                  type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    if st.button('Predict!'):
        show(uploaded_image)
