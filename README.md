# Brain-inspired Incremental Learning

Brain-inspired incremental learning experiments with binary MNIST encodings,
including LTP and MIS learning rules under controlled class interference.

## Project structure

- `mnist_gumbel_embed.py`: generates input encodings from MNIST.
- `input_preprocessing.py`: builds binary input encodings and class-wise
	interference datasets.
- `memory_model.py`: core `MemoryModel` implementation (LTP/MIS learning,
	thresholds selection and weight initialisation).
- `memory_encoding.py`: experiment runner across learning types, interference
	levels, and classes.
- `utils.py`: helper functions.
- `constants.py`: shared constant values.

## Defaults

- Neurons: `60`
- Active neurons per pattern: `12`
- Classes: `0-9`
- Interference levels: `0`, `0.2`, `0.3`, `0.5`
- Class subset size: `100`

## Run

1. Generate processed inputs and interference sets:

```bash
python input_preprocessing.py
```

2. Run memory learning experiments:

```bash
python memory_encoding.py
```