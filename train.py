from os import getenv, path

from models.conv_recognition import ConvRecogntion


# The custom dataset name
DATASET_KEY = 'DATASET'
# The custom model name
MODEL_EXPORT_PATH_KEY = 'MODEL_EXPORT_PATH'
# Whether the model should be optimised
OPTIMISE_KEY = 'OPTIMISE'
# Epochs key
EPOCHS_KEY = 'EPOCHS'
# Patience key
PATIENCE_KEY = 'PATIENCE'


def train(dataset_name: str, model_name: str, optimise: bool,
          epochs: int, patience: int):
    # Create the dataset path
    dataset_path = path.join(path.abspath('datasets'), dataset_name)

    # Create the model
    model = ConvRecogntion(dataset_path, model_name)

    # Build the model
    model.build()
    # Train the model
    model.train(epochs, patience)

    # Write the model
    model.write(optimise)


def main():
    # Get the environment variables
    dataset_name = getenv(DATASET_KEY, 'emotions')
    model_name = getenv(DATASET_KEY, 'emotion_recognition')
    optimise = bool(getenv(DATASET_KEY, True))
    epochs = int(getenv(EPOCHS_KEY, 200))
    patience = int(getenv(PATIENCE_KEY, 5))

    # train the model
    train(dataset_name=dataset_name, model_name=model_name,
          optimise=optimise, epochs=epochs, patience=patience)


if __name__ == "__main__":
    main()
