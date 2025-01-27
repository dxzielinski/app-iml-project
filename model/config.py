"""Configuration file for the model."""


class MasterConfig:
    """
    A class to store all essential configuration constants for the ML project.
    """

    # Paths
    DATA_DIR = "../data-no-noise-no-silence"  # relative path to dataset
    LOGGING_DIRECTORY = "./logs/test-mlflow"  # directory for tensorboard logs
    EXPERIMENT_NAME = "test-mlflow"  # name of experiment for tensorboard logs

    # Model and training
    EPOCHS = 3
    IMG_WIDTH, IMG_HEIGHT = 224, 224  # input image will be resized to this size
    NUM_CLASSES = 30  # adjust to number of classes in dataset
    TASK = "multiclass"
    IN_CHANNELS = 1  # grayscale; 3 for RGB

    # Optimization
    OPTIMIZATION_N_TRIALS = 500
    OPTIMIZATION_TIMEOUT = 600000  # seconds

    # Data preprocessing
    GLOBAL_MIN_DB = -40  # Minimum dB level for consistent scaling
    GLOBAL_MAX_DB = 60  # Maximum dB level for consistent scaling
    SAMPLING_RATE = 44100  # Sampling rate for audio loading
    FRAME_LENGTH = 1024  # Frame length for energy-based silence removal
    HOP_LENGTH = 512  # Hop length for frame-based processing

    CLASS_NAMES = [
        "bed",
        "bird",
        "cat",
        "dog",
        "down",
        "eight",
        "five",
        "four",
        "go",
        "happy",
        "house",
        "left",
        "marvin",
        "nine",
        "no",
        "off",
        "on",
        "one",
        "right",
        "seven",
        "sheila",
        "six",
        "stop",
        "three",
        "tree",
        "two",
        "up",
        "wow",
        "yes",
        "zero",
    ]
