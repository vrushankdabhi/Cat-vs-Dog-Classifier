# Cat vs Dog Classifier

This repository contains the implementation of a Convolutional Neural Network (CNN) for classifying images of cats and dogs. The model is trained on the Cat vs Dog dataset from Kaggle and achieves high accuracy in distinguishing between the two classes.

## Table of Contents

- [Introduction](#introduction)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Prerequisites](#prerequisites)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This project implements a deep learning model to classify images of cats and dogs. Leveraging the power of Convolutional Neural Networks, the model is trained on a large dataset from Kaggle to achieve high accuracy. The dataset used is sourced from the Kaggle Dogs vs Cats dataset, which contains a balanced collection of images of cats and dogs.

## Architecture

The CNN architecture implemented in this project consists of the following layers:

1. **Convolutional Layer 1:** 32 filters, 3x3 kernel, ReLU activation, 'valid' padding
2. **Batch Normalization Layer 1**
3. **Max-Pooling Layer 1:** 2x2 pool size, stride 2, 'valid' padding
4. **Convolutional Layer 2:** 64 filters, 3x3 kernel, ReLU activation, 'valid' padding
5. **Batch Normalization Layer 2**
6. **Max-Pooling Layer 2:** 2x2 pool size, stride 2, 'valid' padding
7. **Convolutional Layer 3:** 128 filters, 3x3 kernel, ReLU activation, 'valid' padding
8. **Batch Normalization Layer 3**
9. **Max-Pooling Layer 3:** 2x2 pool size, stride 2, 'valid' padding
10. **Flatten Layer**
11. **Dense Layer 1:** 128 neurons, ReLU activation
12. **Dropout Layer 1:** 10% dropout rate
13. **Dense Layer 2:** 64 neurons, ReLU activation
14. **Dropout Layer 2:** 10% dropout rate
15. **Output Layer:** 1 neuron, sigmoid activation

## Dataset

The dataset used for training and validation is the [Dogs vs Cats dataset](https://www.kaggle.com/datasets/salader/dogs-vs-cats) from Kaggle. It includes 25,000 images of cats and dogs, divided into training and validation sets.

## Prerequisites

- Python 3.7+
- TensorFlow 2.0+
- Keras
- NumPy
- Matplotlib
- OpenCV
- Kaggle API (for dataset download)

You can install the necessary packages using:

```sh
pip install tensorflow keras numpy matplotlib opencv-python kaggle
```

## Usage

1. **Clone the repository:**

```sh
git clone https://github.com/your-username/Cat-vs-Dog-Classifier.git
cd Cat-vs-Dog-Classifier
```

2. **Download the dataset:**

Ensure you have your Kaggle API token (`kaggle.json`) in the working directory.

```sh
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!kaggle datasets download -d salader/dogs-vs-cats
unzip dogs-vs-cats.zip -d ./data
```

3. **Train the model:**

Run the provided Jupyter notebook or Python script to train the model.

```sh
python train.py
```

4. **Evaluate the model:**

Evaluate the model's performance on the validation set.

```sh
python evaluate.py
```

5. **Visualize training history:**

```sh
python plot_history.py
```

## Results

After training, the model achieves the following performance on the validation set:

- **Accuracy:** 80.77%
The training and validation accuracy and loss over epochs are visualized and saved in the `results` directory.

## Contributing

Contributions are welcome! If you have any suggestions, bug reports, or improvements, feel free to submit an issue or a pull request.
