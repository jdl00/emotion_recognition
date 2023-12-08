# Emotion Recognition Using Convolutional Neural Networks

## Description

This project aims to implement emotion recognition using convolutional neural networks (CNNs), inspired by the architectures and methodologies described in "[Facial Emotion Recognition: State of the Art Performance on FER2013](https://arxiv.org/ftp/arxiv/papers/2105/2105.03588.pdf)" and "[Facial Expression Recognition in the Wild via Deep Attentive Networks](https://iopscience.iop.org/article/10.1088/1742-6596/1844/1/012004/pdf)". The project utilizes TensorFlow and Keras for model development, with a focus on achieving high accuracy in detecting human emotions from facial images.

## Features

- Utilizes a custom-built CNN architecture for emotion recognition.
- Employs Haar cascades for effective face detection in preprocessing.
- Implements data augmentation to enhance model robustness.
- Split data into training and validation sets for effective model evaluation.
- Can be extended to real-time emotion recognition with minimal modifications.

## Model Architecture

The model architecture is based on the principles outlined in the referenced papers, adapted for emotion recognition. Key features include:

- Multiple convolutional layers for feature extraction.
- MaxPooling layers for spatial data reduction.
- Dropout layers to mitigate overfitting.
- Dense layers for classification, with a softmax activation function in the output layer to handle multiple emotion classes.

## Training Dataset

The dataset used for training is the [FER-2013](https://www.kaggle.com/datasets/msambare/fer2013/). It contains images which are categorised to one of 7 categories being _0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral_. The features of the dataset:

- Training set of 28,709 examples in the training set, 3,589 in the testing set
- Faces are already centered and occupies roughly the same amount of space

## Installation

To set up the project, follow these steps:

1. Clone the repository \
   `git clone https://github.com/jdl00/emotion_recognition`

2. Install the required libraries \
   `pip install -r requirements.txt`

## Usage

1. To train the model run the train command

   ```
   # Environment variables

   # Folder to export the model to
   EXPORT MODEL_EXPORT_PATH='folder_to_export_model_run_to'

   # Dataset path to be used (classifiers are pulled from the dataset)
   EXPORT DATASET='dataset_path'

   # By default models are quantised and optimised, to disable this you can export
   EXPORT OPTIMISE=0 (to turn off optimisation)

   # Set the amount of epochs to train for
   EXPORT EPOCHS=200

   # Patience till training early stops
   EXPORT patience=5

   # Finally execute the train script
   'ENV_VARIABLES' python train.py
   ```

2. Perform recognition on camera input

   The realtime input to the model uses Pythons OpenCV library. You can specify which input by using the device_id

   ```
   # Environment variables:

   # Device ID of the camera
   EXPORT DEVICE_ID='device_id_of_camera'

   # Path to the exported model
   EXPORT MODEL_PATH='exported_model_path'

   # Finally execute the recognition script
   'ENV_VARIABLES' python recognition.py
   ```

## Results

`TODO: Complete this section.`

## Contributing

Contributions to this project are welcome. To contribute:

1. Fork the repository.
2. Create a new branch for your feature (`git checkout -b feature/your_feature`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature/your_feature`).
5. Create a new Pull Request.

_Note: Adjust these instructions based on how you wish others to contribute to your project._

## License

[MIT License](LICENSE.md)

