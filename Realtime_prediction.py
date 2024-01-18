# import libraries 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
import cv2 

import keras
from keras.applications.vgg19 import VGG19

import joblib
import matplotlib.pyplot as plt


# Specify the path to the pretrained model file (.h5)
model_path = "models\\my_model_full_dataset.h5"

# Load the model
model = keras.models.load_model(model_path) 

lbl_path = "models\\labels.joblib"
#load labels
lbl_binarizer = joblib.load(lbl_path)

#function for image preprocessing
def prepare_cust_images(img, height = 64, width = 64):
#     images = []
#     for file in os.listdir(filepath):
#         img = cv2.imread(os.path.join(filepath,file))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (height, width))
    img = img / 255.
    images=[img]
#         images.append(img)
    
    return np.array(images)

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame from webcam
    ret, frame = cap.read()

    # Preprocess the frame to match your model's input requirements
#     frame = cv2.resize(frame, (224, 224))  # Adjust the size based on your model's input size
#     frame = frame / 255.0  # Normalize pixel values
    
    
#     # Expand dimensions to match the model's expected input shape
#     frame = np.expand_dims(frame, axis=0)

#     # Make predictions
#     predictions = model.predict(frame)

#     # Get the predicted class label
#     predicted_class = class_labels[np.argmax(predictions)]
    
    new_frame = prepare_cust_images(frame)
    
    predicted_class = model.predict(new_frame)

#     plt.figure(figsize = (25,20))
#     for i in range(len(custom_preds)):
#         plt.subplot(1,2,i+1)
#         plt.imshow(custom_images[i])
#         plt.title('pred: {}'.format(lbl_binarizer.classes_[np.argmax(custom_preds[i], axis = -1)]))

    plt.show()
    # Display the frame and predicted class
    pred = lbl_binarizer.classes_[np.argmax(predicted_class, axis = -1)]
    cv2.putText(frame, f'Prediction: {pred}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Webcam', frame)

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the OpenCV window
cap.release()
cv2.destroyAllWindows()