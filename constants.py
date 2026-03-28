from pathlib import Path

BASE_PATH = Path(__file__).resolve().parent

N_NEURONS_DEFAULT = 60
N_ACTIVATIONS_DEFAULT = 12

UNDEFINED = -1

CLASS_SUBSET_SIZE = 100
MNIST_CLASSES = [i for i in range(10)]
LEARNING_TYPES = ['ltp', 'mis']
INTERFERENCE_LEVELS = [0, 0.2, 0.3, 0.5]

MAX_ITERATIONS = 1000