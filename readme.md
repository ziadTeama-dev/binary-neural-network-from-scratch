# Neural Network Digit Classification with Weight Binarization

A simple **Neural Network implemented from scratch using NumPy** to classify handwritten digits from the `sklearn digits` dataset.

The project includes:

* Forward propagation
* Backpropagation
* ReLU activation function
* Softmax output layer
* L1 and L2 regularization
* Learning rate decay
* Weight clipping
* Optional weight and bias binarization

## Dataset

The model uses the **Digits Dataset** from Scikit-learn.

* Images: 8x8 grayscale handwritten digits
* Classes: 10 (digits from 0 to 9)
* Samples: 1797 images

Example:

```
Input image:
8x8 pixels

Output:
[0,0,0,1,0,0,0,0,0,0]
```

## Model Architecture

The neural network consists of:

```
Input Layer
    |
    | 64 features
    |
Hidden Layer
    |
    | ReLU activation
    |
Output Layer
    |
    | Softmax
    |
10 Classes
```

## Features

### Forward Propagation

The network calculates:

* Hidden layer activation
* ReLU transformation
* Output probabilities using Softmax

### Backpropagation

The gradients are calculated manually using:

* Cross entropy loss
* Weight derivatives
* Bias derivatives

### Regularization

The model supports:

**L1 Regularization**

* Helps remove unnecessary weights

**L2 Regularization**

* Prevents large weights and overfitting

### Weight Binarization

The model supports converting weights and biases into binary representation during forward propagation.

This reduces precision and can simulate binary neural networks.

Example:

Before:

```
[0.34, -0.72, 0.15]
```

After:

```
[1, -1, 1]
```

## Training Results

Without binarization:

```
Accuracy ≈ 97%
```

With binarization:

```
Accuracy ≈ 95%
```

The small accuracy drop is expected because the network uses lower precision weights.

## Requirements

* Python 3.x
* NumPy
* Scikit-learn
* Matplotlib

## Installation

Clone the repository:

```bash
git repo clone ziadTeama-dev/binary-neural-network-from-scratch
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

Run the jupyter-notebook file

The program will:

1. Load the dataset
2. Train the neural network
3. Evaluate accuracy
4. Display misclassified images

## Future Improvements

* Implement Binary Neural Network training instead of only binary inference
* Add mini-batch gradient descent
* Add GPU acceleration
* Compare different binarization methods
* Add model saving/loading

