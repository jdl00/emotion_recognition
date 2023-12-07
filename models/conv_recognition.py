import os

import numpy as np
import tensorflow as tf
import keras
from keras import layers, models
from keras.preprocessing.image import ImageDataGenerator, img_to_array
import cv2


RES = (48, 48)
INPUT_SHAPE = (*RES, 1)


def preprocess(image, face_cascade, target_size):

    gray_image = image
    try:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    except cv2.error:
        pass
    except Exception as e:
        print(e)

    faces = face_cascade.detectMultiScale(
        gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        face = gray_image[y:y+h, x:x+w]
        face = cv2.resize(face, target_size)
        face = img_to_array(face)
        face = np.expand_dims(face, axis=0)
        face /= 255.0
        return face
    else:
        # Adjusted to match the target_size and channel
        return np.zeros((*target_size, 1))


class ConvRecogntion:

    def __init__(self, dataset_path: str) -> None:
        self.__dataset_path = dataset_path
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.__model = None

    def __create_dataset(self):
        def preprocess_lambda(x): return preprocess(x, self.face_cascade, RES)

        data_gen = ImageDataGenerator(rescale=1./255,
                                      preprocessing_function=preprocess_lambda)

        # Load and preprocess the training dataset
        self.__data_generator = data_gen.flow_from_directory(
            self.__dataset_path,
            target_size=RES,
            batch_size=64,
            class_mode='categorical',
            color_mode='grayscale'
        )

    def build(self):
        # Build the model
        self.__model = models.Sequential([
            layers.Input(shape=INPUT_SHAPE),
            layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            layers.Flatten(),
            layers.Dense(3000, activation='relu'),
            layers.Dense(7, activation='softmax')
        ])

        self.__model.compile(optimizer='adam',
                             loss='mean_squared_error',
                             metrics=['accuracy'])

        # Display the model structure
        self.__model.summary()

    def train(self):
        ...
