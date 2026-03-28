import math

import numpy as np
from pathlib import Path

from constants import BASE_PATH, UNDEFINED, LEARNING_TYPES, MAX_ITERATIONS

from utils import _validate_binary_inputs


class MemoryModel:
    """Fully connected bipartite network with custom LTP/MIS learning procedures."""

    def __init__(self, n_neurons, n_activations, weights):
        """Initialize the memory model.

        Args:
            n_neurons: Number of neurons in both input and output layers.
            n_activations: Expected number of active neurons per input and output encoding.
            weights: Initial weight matrix with shape `(n_neurons, n_neurons)`.

        Raises:
            ValueError: If neuron counts are invalid or weights are not 2D.
        """
        if n_neurons <= 0:
            raise ValueError("n_neurons must be greater than 0")
        if n_activations < 0 or n_activations > n_neurons:
            raise ValueError("n_activations must be between 0 and n_neurons")
        if weights.ndim != 2:
            raise ValueError("weights must be a 2D numpy matrix")

        self.n_neurons = n_neurons
        self.n_activations = n_activations
        self.weights = weights.astype(float)
        self.learning_result = None

        if self.weights.shape != (self.n_neurons, self.n_neurons):
            raise ValueError(
                f"weights must have shape ({self.n_neurons}, {self.n_neurons})"
            )

    def learn(
        self,
        input_sequence,
        learning_type="ltp",
        learning_rate=1.2,
        threshold=0.2621,
        threshold_step=0.001,
        learnable_weights_filter=None,
    ):
        """Run the learning process sequentially for all inputs.

        Args:
            input_sequence: Binary numpy.ndarray of input encodings, each with shape `(n_neurons,)`.
            learning_type: Learning rule to apply (`"ltp"` or `"mis"`).
            learning_rate: Multiplicative update factor used by LTP.
            threshold: Initial activation threshold.
            threshold_step: Threshold increment applied after each input.
            learnable_weights_filter: Optional binary mask with same shape as
                weights indicating which weights can be modified.

        Raises:
            ValueError: If inputs, learning type, or filter shape are invalid.
        """

        if learning_type not in LEARNING_TYPES:
            raise ValueError("Invalid learning type")

        if input_sequence.ndim != 2:
            raise ValueError(
                "input_sequence must be a 2D matrix of shape (n_neurons, n_inputs)"
            )
        _validate_binary_inputs(input_sequence, self.n_neurons)

        if learnable_weights_filter is not None:
            if learnable_weights_filter.shape != self.weights.shape:
                raise ValueError(
                    f"learnable_weights_filter must have shape {self.weights.shape}"
                )

        weights = self.weights.copy()
        current_threshold = float(threshold)
        n_inputs = input_sequence.shape[1]

        iteration_counts = np.full(n_inputs, UNDEFINED, dtype=int)
        pre_activation_counts = np.full(n_inputs, UNDEFINED, dtype=int)
        all_active_output_neurons = np.full(
            (self.n_activations, n_inputs),
            UNDEFINED,
            dtype=int,
        )
        output_activation_sequence = np.full((self.n_neurons, n_inputs), UNDEFINED, dtype=int)

        for input_idx in range(n_inputs):
            input_encoding = input_sequence[:, input_idx]
            iteration_count = 0
            pre_activation_count = 0
            active_output_neurons = np.array([], dtype=int)
            pre_activated_neurons = np.array([], dtype=int)

            pre_output_signals = np.dot(weights.T, input_encoding)
            pre_activated_neurons = np.where(pre_output_signals >= current_threshold)[0]
            pre_activation_count = pre_activated_neurons.shape[0]

            if pre_activation_count > self.n_activations:
                selected_indices = np.argpartition(
                    pre_output_signals[pre_activated_neurons],
                    -self.n_activations,
                )[-self.n_activations:]
                pre_activated_neurons = pre_activated_neurons[selected_indices]

            if learning_type == 'ltp':
                weights, iteration_count, active_output_neurons = self._execute_ltp(
                    weights,
                    input_encoding,
                    learning_rate,
                    current_threshold,
                    pre_activated_neurons,
                    learnable_weights_filter=learnable_weights_filter,
                )
                weights = self._stabilise_weights(weights)

            elif learning_type == 'mis':
                weights, iteration_count, active_output_neurons = self._execute_mis(
                    weights,
                    input_encoding,
                    current_threshold,
                    pre_activated_neurons,
                    learnable_weights_filter=learnable_weights_filter,
                )
                
            current_threshold += threshold_step
            active_output_neurons = np.asarray(active_output_neurons, dtype=int)

            if len(active_output_neurons) > self.n_activations:
                raise ValueError(
                    f"Input {input_idx}: got {len(active_output_neurons)} activated neurons, "
                    f"but n_activations={self.n_activations}"
                )

            all_active_output_neurons[:, input_idx] = active_output_neurons
            iteration_counts[input_idx] = iteration_count
            pre_activation_counts[input_idx] = pre_activation_count
            pre_activated_neurons = np.asarray(pre_activated_neurons, dtype=int)

            # Mark pre-activated as 1, post-activated as 2 for identification
            for neuron in active_output_neurons:
                if neuron != UNDEFINED:
                    if neuron in pre_activated_neurons:
                        output_activation_sequence[neuron, input_idx] = 1
                    else:
                        output_activation_sequence[neuron, input_idx] = 2

        self.weights = weights

        self.learning_result = {
            "learning_type": learning_type,
            "iteration_counts": iteration_counts.copy(),
            "pre_activation_counts": pre_activation_counts.copy(),
            "active_output_neurons": all_active_output_neurons.copy(),
            "output_activation_sequence": output_activation_sequence.copy(),
            "final_weights": self.weights.copy(),
            "final_threshold": float(current_threshold),
        }

    def save_learning_results(self, result_dir="results", tags=None):
        """Save outputs from the latest learning run.

        Args:
            result_dir: Output directory path.
            tags: Optional sequence of strings appended to output filenames.

        Raises:
            ValueError: If no learning result is available.
        """
        tags = list(tags) if tags is not None else []
        tags_str = f"_{'_'.join(tags)}" if tags else ""

        result = self.learning_result
        if result is None:
            raise ValueError("No learning result available. Run learn() first.")

        learning_type = result["learning_type"]
        output_path = Path(result_dir)
        if not output_path.is_absolute():
            output_path = BASE_PATH / output_path
        output_path.mkdir(parents=True, exist_ok=True)

        np.savetxt(
            output_path / f"{learning_type}_iteration_counts{tags_str}.csv",
            result["iteration_counts"],
            delimiter=",",
            fmt="%d",
        )
        np.savetxt(
            output_path / f"{learning_type}_pre_activation_counts{tags_str}.csv",
            result["pre_activation_counts"],
            delimiter=",",
            fmt="%d",
        )
        np.savetxt(
            output_path / f"{learning_type}_active_output_neurons{tags_str}.csv",
            result["active_output_neurons"],
            delimiter=",",
            fmt="%d",
        )
        np.savetxt(
            output_path / f"{learning_type}_output_activation_sequence{tags_str}.csv",
            result["output_activation_sequence"],
            delimiter=",",
            fmt="%d",
        )

    @staticmethod
    def _stabilise_weights(weights):
        """Rescale each input row so its outgoing weights sum to one.

        Args:
            weights: Weight matrix.

        Returns:
            Row-wise normalized weight matrix.
        """
        rescaled_weights = np.zeros_like(weights)
        for i in range(weights.shape[0]):
            row_sum = np.sum(weights[i, :])
            if row_sum == 0:
                rescaled_weights[i, :] = weights[i, :]
            else:
                rescaled_weights[i, :] = weights[i, :] / row_sum

        return rescaled_weights


    def _execute_ltp(
        self,
        weights,
        input_encoding,
        learning_rate,
        threshold,
        pre_activated_neurons,
        learnable_weights_filter=None,
    ):
        """Apply one LTP update pass for a single input.

        Args:
            weights: Current weight matrix.
            input_encoding: Binary input vector.
            learning_rate: Multiplicative update factor.
            threshold: Activation threshold.
            pre_activated_neurons: Indices already activated before learning.
            learnable_weights_filter: Optional mask indicating modifiable weights.

        Returns:
            Tuple `(updated_weights, iteration_count, active_output_neurons)`.
        """
        weights = weights.copy()

        if learnable_weights_filter is not None:
            learnable_weights_filter = learnable_weights_filter.copy()
        else:
            learnable_weights_filter = np.ones_like(weights)

        iteration_count = 0
        active_output_neurons = np.full(self.n_activations, UNDEFINED, dtype=int)
        output_signals = np.dot(weights.T, input_encoding)

        active_output_neurons_count = pre_activated_neurons.shape[0]
        if active_output_neurons_count > 0:
            active_output_neurons[:active_output_neurons_count] = pre_activated_neurons

        while active_output_neurons_count < self.n_activations:
            if iteration_count >= MAX_ITERATIONS:
                break

            selected_output_neuron_idx = self._roulette_wheel_selection(output_signals)

            if selected_output_neuron_idx in active_output_neurons:
                continue

            active_edges = np.where(
                (input_encoding != 0)
                & (learnable_weights_filter[:, selected_output_neuron_idx] != 0)
            )[0]
            weights[active_edges, selected_output_neuron_idx] *= learning_rate
            output_signals[selected_output_neuron_idx] = np.dot(
                weights[:, selected_output_neuron_idx],
                input_encoding,
            )

            if output_signals[selected_output_neuron_idx] >= threshold:
                active_output_neurons[active_output_neurons_count] = selected_output_neuron_idx
                active_output_neurons_count += 1
                learnable_weights_filter[active_edges, selected_output_neuron_idx] = 0

            iteration_count += 1

        return weights, iteration_count, active_output_neurons


    def _execute_mis(
        self,
        weights,
        input_encoding,
        threshold,
        pre_activated_neurons,
        learnable_weights_filter=None,
    ):
        """Apply one MIS update pass for a single input.

        Args:
            weights: Current weight matrix.
            input_encoding: Binary input vector.
            threshold: Activation threshold.
            pre_activated_neurons: Indices already activated before MIS rewiring.
            learnable_weights_filter: Optional mask indicating modifiable weights.

        Returns:
            Tuple `(updated_weights, iteration_count, active_output_neurons)`.
        """

        if learnable_weights_filter is not None:
            learnable_weights_filter = learnable_weights_filter.copy()
        else:
            learnable_weights_filter = np.ones_like(weights)

        weights = weights.copy()
        learnable_weights = weights * learnable_weights_filter

        iteration_count = 0
        active_output_neurons = np.full(self.n_activations, UNDEFINED, dtype=int)
        excluded_neurons = []
        output_signals = np.dot(weights.T, input_encoding)

        active_output_neurons_count = pre_activated_neurons.shape[0]
        if active_output_neurons_count > 0:
            active_output_neurons[:active_output_neurons_count] = pre_activated_neurons

        while active_output_neurons_count < self.n_activations:
            if iteration_count >= MAX_ITERATIONS:
                break

            # Set up so that only output neurons that are not already active or excluded can be selected by the roulette wheel selection
            candidate_output_signals = [
                output_signals[i]
                if (i not in active_output_neurons and i not in excluded_neurons)
                else 0
                for i in range(self.n_neurons)
            ]

            # If all remaining candidate output neurons have zero signal, or if all output neurons are either active or excluded, end the learning process
            if (
                len(excluded_neurons) + active_output_neurons_count >= self.n_neurons
                or np.sum(candidate_output_signals) == 0
            ):
                break

            selected_output_neuron_idx = self._roulette_wheel_selection(candidate_output_signals)

            if selected_output_neuron_idx in active_output_neurons or selected_output_neuron_idx in excluded_neurons:
                continue

            incoming_edges = learnable_weights[:, selected_output_neuron_idx] * input_encoding
            active_edge_indices = np.where(incoming_edges != 0)[0]

            if len(active_edge_indices) == 0:
                excluded_neurons.append(selected_output_neuron_idx)
                continue

            # Select a source edge to rewire from (one of the active edges feeding this output neuron)
            chosen_edge_idx = self._roulette_wheel_selection(
                weights[active_edge_indices, selected_output_neuron_idx]
            )
            chosen_edge_input_idx = active_edge_indices[chosen_edge_idx]
            
            # Randomly choose direction (0=forward, 1=backward) for searching donor neuron
            direction = np.random.randint(0, 2)
            offset = 0

            # Search for a free output neuron to rewire the weight edge to
            while offset < len(weights[chosen_edge_input_idx, :]):

                if offset == len(weights[chosen_edge_input_idx, :]) - 1:
                    # Searched all output neurons- exclude this one from future attempts
                    excluded_neurons.append(selected_output_neuron_idx)
                    break

                offset += 1

                # Compute candidate output neuron index based on random direction
                if direction == 0:
                    next_edge_index = (selected_output_neuron_idx + offset) % len(weights[chosen_edge_input_idx, :])
                else:
                    next_edge_index = (selected_output_neuron_idx - offset) % len(weights[chosen_edge_input_idx, :])

                # Skip if selected edge is locked (already rewired, part of active neuron, or not learnable)
                if learnable_weights[chosen_edge_input_idx, next_edge_index] == 0:
                    continue

                weights[chosen_edge_input_idx, selected_output_neuron_idx] += weights[chosen_edge_input_idx, next_edge_index]
                weights[chosen_edge_input_idx, next_edge_index] = 0

                # Both edges involved in the re-wiring are no longer learnable for the current input
                learnable_weights_filter[
                    chosen_edge_input_idx,
                    selected_output_neuron_idx,
                ] = 0
                learnable_weights_filter[
                    chosen_edge_input_idx,
                    next_edge_index,
                ] = 0
                learnable_weights = weights * learnable_weights_filter
                output_signals = np.dot(weights.T, input_encoding)
                iteration_count += 1

                if output_signals[selected_output_neuron_idx] >= threshold:
                    active_output_neurons[active_output_neurons_count] = selected_output_neuron_idx
                    active_output_neurons_count += 1
                    learnable_weights_filter[:, selected_output_neuron_idx] = 0
                    learnable_weights = weights * learnable_weights_filter
                    break

        return weights, iteration_count, active_output_neurons


    def _roulette_wheel_selection(self, probabilities, inverse=False, n=1):
        """Select one or more indices using roulette-wheel sampling.

        Args:
            probabilities: Sequence of selection weights.
            inverse: If True, uses inverse weighting before normalisation.
            n: Number of unique selections.

        Returns:
            Selected index when `n == 1`, otherwise a list of indices.

        Raises:
            ValueError: If all probabilities are zero after preprocessing.
        """
        # Scale negative probabilities to positive
        if np.min(probabilities) < 0:
            probabilities = probabilities - np.min(probabilities) + 1e-6

        # Invert probabilities if needed
        if inverse:
            probabilities = 1 / probabilities

        # Normalise probabilities (if not already normalized)
        total_sum = np.sum(probabilities)

        if total_sum == 0:
            raise ValueError("Sum of probabilities is zero. Cannot perform selection.")

        probabilities = probabilities / total_sum
        selected_indices = []

        for _ in range(n):
            cumulative_probabilities = np.cumsum(probabilities)
            random_number = np.random.rand()

            for i, cumulative_probability in enumerate(cumulative_probabilities):
                if random_number < cumulative_probability:
                    selected_indices.append(i)

                    # Set the probability of the selected index to zero to prevent it from being selected again
                    probabilities[i] = 0
                    # Normalize probabilities again
                    total_sum = np.sum(probabilities)
                    if total_sum > 0:
                        probabilities = probabilities / total_sum
                    break

        return selected_indices if n > 1 else selected_indices[0]


def initialise_weights(n_neurons):
    """Initialize and row-normalize a random weight matrix.

    Args:
        n_neurons: Number of neurons in each layer.

    Returns:
        Weight matrix with shape `(n_neurons, n_neurons)` where each row
        sums to 1.

    Raises:
        ValueError: If `n_neurons` is not greater than 0.
    """

    if n_neurons <= 0:
        raise ValueError("n_neurons must be greater than 0")

    weights = np.zeros((n_neurons, n_neurons), dtype=float)

    for neuron_idx in range(n_neurons):
        row_weights = 0.2 * np.random.randn(n_neurons) + 1
        row_weights /= np.sum(row_weights)
        weights[neuron_idx, :] = row_weights

    return weights


def select_threshold(inputs, weights):
    """Compute a conservative threshold above all current output activations.

    Args:
        inputs: Input pattern matrix with shape
            `(n_neurons, n_inputs)`.
        weights: Weight matrix with shape `(n_neurons, n_neurons)`.

    Returns:
        Threshold rounded up to 4 decimal places.
    """

    max_outputs = np.max(np.dot(weights.T, inputs), axis=0)
    threshold = np.max(max_outputs) * 1.000001

    return math.ceil(threshold * 10000) / 10000
