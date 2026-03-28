import numpy as np

from memory_model import MemoryModel, select_threshold

from constants import (
    LEARNING_TYPES,
    INTERFERENCE_LEVELS,
    MNIST_CLASSES,
    N_NEURONS_DEFAULT,
    N_ACTIVATIONS_DEFAULT,
)
from utils import _data_path, _validate_binary_inputs, _load_or_initialise_weights

def main():
    """Run memory encoding across classes and interference levels.

    Loads initial weights and class-specific input encodings, computes a global
    threshold once from the original encodings, then runs learning for each
    configured learning type and interference level.

    Raises:
        FileNotFoundError: If required input files are missing.
        ValueError: If loaded or generated data fails validation checks.
    """
    weights_path = _data_path('initial_weights_60_neurons.csv')
    base_inputs_path = _data_path('mnist_input_encodings.csv')

    if not base_inputs_path.exists():
        raise FileNotFoundError(f"Input encodings file not found: {base_inputs_path}")

    weights = _load_or_initialise_weights(weights_path)
    inputs = np.loadtxt(base_inputs_path, delimiter=',').astype(int)

    _validate_binary_inputs(inputs, N_NEURONS_DEFAULT)

    # Global threshold selection based on the original input encodings
    threshold = select_threshold(inputs, weights)

    for learning_type in LEARNING_TYPES:
        for interference in INTERFERENCE_LEVELS:
            for class_label in MNIST_CLASSES:
                inputs_path = _data_path(
                    f'mnist_input_encodings_class_{class_label}_interference_{interference * 100:.0f}.csv'
                )

                if not inputs_path.exists():
                    raise FileNotFoundError(
                        f"Input file not found: {inputs_path}. Generate the subset "
                        "encodings with input_preprocessing.py before learning."
                    )

                inputs = np.loadtxt(inputs_path, delimiter=',').astype(int)
                _validate_binary_inputs(inputs, N_NEURONS_DEFAULT)

                model = MemoryModel(
                    n_neurons=N_NEURONS_DEFAULT,
                    n_activations=N_ACTIVATIONS_DEFAULT,
                    weights=weights,
                )
                model.learn(inputs, learning_type=learning_type, threshold=threshold)
                model.save_learning_results(
                    'results',
                    tags=[
                        f'class_{class_label}',
                        f'interference_{interference * 100:.0f}',
                    ],
                )



if __name__ == "__main__":
    main()
