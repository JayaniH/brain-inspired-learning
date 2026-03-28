import numpy as np
import pandas as pd

from constants import N_NEURONS_DEFAULT, N_ACTIVATIONS_DEFAULT, INTERFERENCE_LEVELS, CLASS_SUBSET_SIZE
from utils import _data_path


def sort_neuron_columns(data_df):
    """Sort neuron columns while preserving labels.

    Args:
        data_df: Input DataFrame containing neuron columns and label.

    Returns:
        DataFrame with sorted neuron columns.
    """

    neuron_cols = [col for col in data_df.columns if 'neuron' in col]
    structured_data = data_df[neuron_cols].copy()
    structured_data = structured_data.apply(lambda x: pd.Series(sorted(x)), axis=1)
    structured_data['label'] = data_df['label']

    return structured_data


def remove_duplicates_within_classes(data_df):
    """Remove duplicate rows from the dataset.

    Args:
        data_df: Input DataFrame.

    Returns:
        DataFrame with duplicated rows removed.
    """

    data_df = data_df.copy()
    return data_df.drop_duplicates()


def remove_duplicates_between_classes(data_df):
    """Remove inputs that occur across multiple classes.

    Args:
        data_df: Input DataFrame.

    Returns:
        DataFrame with rows removed when their active neurons are duplicated
        across classes.
    """

    data_df = data_df.copy()
    subset_cols = [col for col in data_df.columns if col != 'label']
    mask = data_df.duplicated(subset=subset_cols, keep=False)
    return data_df[~mask]


def generate_input_encodings(
    data_df,
    n_neurons=N_NEURONS_DEFAULT,
    n_activations=N_ACTIVATIONS_DEFAULT,
):
    """Create binary neuron encodings from selected active neuron indices.

    Args:
        data_df: Input DataFrame where the first `n_activations` columns contain
            active neuron indices per input.
        n_neurons: Total number of neurons in each encoding vector.
        n_activations: Number of active neurons expected per input.

    Returns:
        Binary array of shape `(n_neurons, n_inputs)`.
    """

    data_df = data_df.copy()
    input_count = data_df.shape[0]
    neuron_cols = list(range(n_activations))
    input_encodings = np.zeros((n_neurons, input_count), dtype=int)

    for i, row in enumerate(data_df[neuron_cols].itertuples(index=False, name=None)):
        active_neurons = [int(n) for n in row if pd.notna(n)]
        input_encodings[active_neurons, i] = 1

    return input_encodings


def get_distance(x, y):
    """Compute overlap-based distance between two binary vectors.

    Args:
        x: First binary vector.
        y: Second binary vector.

    Returns:
        Distance value where lower means more overlap.
    """

    return 1 - np.sum(np.logical_and(x, y)) / N_ACTIVATIONS_DEFAULT


def select_class_subsets(input_encodings, labels, num_per_class):
    """Select representative samples per class using a prototype distance.

    Args:
        input_encodings: Binary numpy.ndarray of shape `(n_neurons, n_inputs)`.
        labels: Class labels with shape `(n_inputs,)`.
        num_per_class: Number of samples to retain for each class.

    Returns:
        List of selected global sample indices.
    """

    selected_idx = []

    for label in np.unique(labels):
        class_idx = np.where(labels == label)[0]
        class_encodings = input_encodings[:, class_idx]

        prototype = np.zeros(input_encodings.shape[0], dtype=int)
        activation_counts = np.sum(class_encodings, axis=1)
        top_positions = np.argsort(activation_counts)[-N_ACTIVATIONS_DEFAULT:]
        prototype[top_positions] = 1

        distances = np.array(
            [get_distance(class_encodings[:, i], prototype) for i in range(class_encodings.shape[1])]
        )

        closest_indices = np.argsort(distances)[:num_per_class]
        selected_class_indices = class_idx[closest_indices]
        selected_idx.extend(selected_class_indices.tolist())

    return selected_idx


def create_input_sets_with_interference(
    input_encodings,
    labels,
    interference_percentage=0.5,
    random_seed=42,
):
    """Create and save class-wise input sets with controlled interference.

    For each class, a fraction of in-class inputs is replaced by out-of-class
    inputs and both encodings and labels are saved to disk.

    Args:
        input_encodings: Binary numpy.ndarray of shape `(n_neurons, n_inputs)`.
        labels: Label array with shape `(n_inputs,)`.
        interference_percentage: Fraction of each class set replaced by inputs
            sampled from other classes.
        random_seed: Seed used for deterministic sampling.
    """

    labels = np.asarray(labels)
    unique_labels = np.unique(labels)
    np.random.seed(random_seed)

    for label in unique_labels:
        class_mask = labels == label
        class_encodings = input_encodings[:, class_mask]
        inputs_with_interference = class_encodings.copy()
        labels_for_class = np.full(class_encodings.shape[1], label, dtype=labels.dtype)

        n_encodings = inputs_with_interference.shape[1]
        n_interference_inputs = int(n_encodings * interference_percentage)

        if n_interference_inputs > 0:
            idx_to_replace = np.random.choice(n_encodings, n_interference_inputs, replace=False)

            out_of_class_indices = np.where(~class_mask)[0]
            allow_replacement = out_of_class_indices.shape[0] < n_interference_inputs
            sampled_out_idx = np.random.choice(
                out_of_class_indices.shape[0],
                n_interference_inputs,
                replace=allow_replacement,
            )
            sampled_global_indices = out_of_class_indices[sampled_out_idx]

            interference_encodings = input_encodings[:, sampled_global_indices]
            interference_labels = labels[sampled_global_indices]

            inputs_with_interference[:, idx_to_replace] = interference_encodings
            labels_for_class[idx_to_replace] = interference_labels

        np.savetxt(
            _data_path(
                f'mnist_input_encodings_class_{label}_interference_{interference_percentage * 100:.0f}.csv'
            ),
            inputs_with_interference.astype(int),
            delimiter=',',
            fmt='%d',
        )
        np.savetxt(
            _data_path(
                f'mnist_input_labels_class_{label}_interference_{interference_percentage * 100:.0f}.csv'
            ),
            labels_for_class,
            delimiter=',',
            fmt='%d',
        )


def main():
    """Run the full preprocessing pipeline and save generated datasets."""

    data_df = pd.read_csv(_data_path('mnist_idx_train.csv'))

    sorted_data_df = sort_neuron_columns(data_df)
    unique_within_classes_df = remove_duplicates_within_classes(sorted_data_df)
    unique_between_classes_df = remove_duplicates_between_classes(unique_within_classes_df)

    input_encodings = generate_input_encodings(unique_between_classes_df)

    unique_between_classes_df.to_csv(_data_path('mnist_idx_train_processed.csv'), index=False)
    np.savetxt(_data_path('mnist_input_encodings.csv'), input_encodings.astype(int), delimiter=',', fmt='%d')

    class_subset_idx = select_class_subsets(
        input_encodings,
        unique_between_classes_df['label'].values,
        num_per_class=CLASS_SUBSET_SIZE,
    )
    subset_input_encodings = input_encodings[:, class_subset_idx]
    np.savetxt(
        _data_path('mnist_input_encodings_subsets.csv'),
        subset_input_encodings.astype(int),
        delimiter=',',
        fmt='%d',
    )

    subsets_df = unique_between_classes_df.iloc[class_subset_idx]
    subsets_df.to_csv(_data_path('mnist_idx_train_subsets.csv'), index=False)

    for interference_percentage in INTERFERENCE_LEVELS:
        create_input_sets_with_interference(
            subset_input_encodings,
            subsets_df['label'].values,
            interference_percentage=interference_percentage,
        )


if __name__ == "__main__":
    main()
