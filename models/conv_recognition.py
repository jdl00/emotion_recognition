import os

import numpy as np
import tensorflow as tf
from keras import layers, models
from keras.preprocessing.image import ImageDataGenerator, img_to_array
from keras.callbacks import EarlyStopping, TensorBoard
from keras.optimizers.legacy import SGD
import cv2


RES = (48, 48)
INPUT_SHAPE = (*RES, 1)
HAAR_CASCADE_PATH = os.path.join(os.path.abspath(
    'open_cv_model'), 'haarcascade_frontalface_default.xml')


def preprocess(image: np.array,
               face_cascade: cv2.CascadeClassifier) -> np.array:
    """Preprocesses the images and selects the features

    Args:
        image (np.array): The array of the image
        face_cascade (cv2.CascadeClassifier): Face classifier for features

    Returns:
        np.array: The selected features
    """
    image = image.astype(np.uint8)

    try:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    except cv2.error:
        gray_image = image
    except Exception:
        assert False

    faces = face_cascade.detectMultiScale(
        gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        face = gray_image[y:y+h, x:x+w]
        face = cv2.resize(face, RES)
        face = img_to_array(face)
        face = np.expand_dims(face, axis=0)
        face /= 255.0
        print(f"Change face {face}")
        print(f"Original face {image}")
        return face
    else:
        # Adjusted to match the target_size and channel
        return image.astype(np.float64)


class ConvRecogntion:
    """Creates an instance of the convolutional model
    """

    def __init__(self, dataset_path: str, model_name: str) -> None:
        """Creates an instance of the model

        Args:
            dataset_path (str): The path to the dataset to be used
            model_name (str): The name of the model to be output
        """
        self.__dataset_path = dataset_path
        self.__face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
        self.__model_name = model_name

    def __create_dataset(self) -> None:
        """Creates the data generators of test/validation
        """

        train_path = os.path.join(self.__dataset_path, 'train')
        test_path = os.path.join(self.__dataset_path, 'test')

        def preprocess_lambda(x): return preprocess(x, self.__face_cascade)

        data_gen = ImageDataGenerator(rescale=1./255)

        # Load and preprocess the training dataset
        self.__train_data_gen = data_gen.flow_from_directory(
            train_path,
            target_size=RES,
            batch_size=64,
            class_mode='categorical',
            color_mode='grayscale'
        )

        # Load and preprocess the training dataset
        self.__test_data_gen = data_gen.flow_from_directory(
            test_path,
            target_size=RES,
            batch_size=64,
            class_mode='categorical',
            color_mode='grayscale'
        )

        # Classes for the final dense layer output
        self.__classes = len(self.__train_data_gen.class_indices)

    def __representative_dataset(self):
        """Creates a representitive dataset for quantisations

        Yields:
            float: Representation of the dataset
        """

        for input_data, _ in self.__train_data_gen:
            for i in range(100):
                yield [input_data]
                break

    def build(self):
        """Builds/Compiles the model and the dataset
        """

        # Create the datasets
        self.__create_dataset()

        self.__model = models.Sequential([
            # First Conv Block
            layers.Conv2D(64, (3, 3), padding='same', input_shape=(48, 48, 1)),
            layers.Activation('relu'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), padding='same'),
            layers.Activation('relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.25),

            # Second Conv Block
            layers.Conv2D(128, (3, 3), padding='same'),
            layers.Activation('relu'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), padding='same'),
            layers.Activation('relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.25),

            # Third Conv Block
            layers.Conv2D(256, (3, 3), padding='same'),
            layers.Activation('relu'),
            layers.BatchNormalization(),
            layers.Conv2D(256, (3, 3), padding='same'),
            layers.Activation('relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.25),

            # Flatten and Fully Connected Layers
            layers.Flatten(),
            layers.Dense(1024),
            layers.Activation('relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),

            layers.Dense(self.__classes, activation='softmax')
        ])

        sgd_optimiser = SGD(learning_rate=0.0001,
                            momentum=0.9)

        # Compile the model with adam and metrics
        self.__model.compile(optimizer=sgd_optimiser,
                             loss='categorical_crossentropy',
                             metrics=['accuracy'])

        # Display the model structure
        self.__model.summary()

    def train(self, epochs: int, patience: int):
        """Trains the model on the dataset

        Args:
            epochs (int): The amount of epochs to run
            patience (int): How many epochs before early stopping
        """

        # Define the tensorboard callback and early stopping callback
        tensorboard_callback = TensorBoard(log_dir="./logs")
        early_stopping_callback = EarlyStopping(monitor='val_loss',
                                                patience=patience,
                                                restore_best_weights=True)

        # Calculate the ideal steps based on the sample and batch size etc
        validation_steps = (self.__test_data_gen.samples //
                            self.__test_data_gen.batch_size)
        steps_per_epoch = (self.__train_data_gen.samples //
                           self.__train_data_gen.batch_size)

        # Train the model
        self.__model.fit(
            self.__train_data_gen,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=self.__test_data_gen,
            validation_steps=validation_steps,
            callbacks=[tensorboard_callback, early_stopping_callback]
        )

    def write(self, quantise: bool):
        """Saves the trained model to a file.

        Args:
            quantize (bool): Whether any optimizations should be done
        """

        # Create a converter
        converter = tf.lite.TFLiteConverter.from_keras_model(self.__model)

        # Apply quantization if desired
        if quantise:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]

            converter.representative_dataset = self.__representative_dataset

            # Ensure that if ops can't be quantized, the converter throws error
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

            # Set the input and output tensors to uint8
            converter.inference_input_type = tf.uint8
            converter.inference_output_type = tf.uint8

        # Convert the model
        tflite_model = converter.convert()

        # Save the model to a file
        with open(f"{self.__model_name}.tflite", 'wb') as f:
            f.write(tflite_model)
