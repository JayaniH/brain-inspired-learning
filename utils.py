import numpy as np

from constants import BASE_PATH, N_NEURONS_DEFAULT

def _data_path(filename):
    """Build a path under the repository data directory.

    Args:
        filename: Name of the file in the `data` directory.

    Returns:
        Absolute path to the file.
    """

    return BASE_PATH / 'data' / filename


def _load_or_initialise_weights(weights_path):
    """Load weights from disk or initialize and save them if missing.

    Args:
        weights_path: Path to the weights CSV file.

    Returns:
        Weight matrix with shape `(N_NEURONS_DEFAULT, N_NEURONS_DEFAULT)`.

    Raises:
        ValueError: If loaded or generated weights have an invalid shape.
    """

    if weights_path.exists():
        weights = np.loadtxt(weights_path, delimiter=",")
    else:
        from memory_model import initialise_weights

        weights = initialise_weights(n_neurons=N_NEURONS_DEFAULT)
        np.savetxt(weights_path, weights, delimiter=",", fmt="%.10f")

    if weights.shape != (N_NEURONS_DEFAULT, N_NEURONS_DEFAULT):
        raise ValueError(
            f"Weights must have shape ({N_NEURONS_DEFAULT}, {N_NEURONS_DEFAULT}) but got {weights.shape}"
        )

    return weights


def _validate_binary_inputs(inputs, n_neurons):
    """Validate input encoding shape and binary values.

    Args:
        inputs: Input encoding matrix.
        n_neurons: Expected neuron count.

    Raises:
        ValueError: If shape is invalid or values are not binary.
    """

    if inputs.shape[0] != n_neurons:
        raise ValueError(
            f"all input encodings must have shape ({n_neurons}, ) but got ({inputs.shape[0]}, )"
        )

    if not np.all((inputs == 0) | (inputs == 1)):
        raise ValueError("inputs must be binary (0/1)")
