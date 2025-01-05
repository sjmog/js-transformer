# Challenge 6: Building a Position-Wise Feedforward Layer

## Title & Introduction

We've tokenized and embedded our input strings, and performed multi-head self-attention. The next step of the Transformer block is to apply a **feedforward network** to the output of the self-attention component.

The type of feedforward network used in a Transformer block is a **Multi-Layer Perceptron (MLP)**, which is a neural network with one or more hidden layers.

![A feedforward network with a single "hidden layer"](https://upload.wikimedia.org/wikipedia/commons/5/54/Feed_forward_neural_net.gif)

In the image above, the feedforward network has a single "hidden layer", containing 4 nodes, which we call **neurons**. The `hiddenDim`, or "hidden dimensions", is 4.

> Usually, the number of neurons in the hidden layer is much larger the number of neurons in the input layer. This is because the core of a neural network is to project the input into a higher-dimensional space, do some transformations, and then project it back down to the original space.

We need to apply this MLP identically to each position in the output of the self-attention component. We call this **position-wise** application.

So, we're going to build a **position-wise feedforward network**.

## Learning Objectives

- Implement a position-wise feedforward network as a 2-layer MLP (linear transformation → ReLU → linear transformation).

> As we saw in the previous challenge, "linear transformation" means "matrix multiplication", although this time we're also adding a **bias** vector to the result.

### To complete this challenge, you will need to:

1. Write a function, `feedForward`, that takes in an input matrix and trainable parameters `W_1`, `b_1`, `W_2`, `b_2`.
  * `W_1` and `W_2` have the shape `[inputDimensions, hiddenDimensions]`.
  * `b_1` and `b_2` have the shape `[hiddenDimensions]`.
2. Transform the input matrix by multiplying with the trainable matrix `W_1` and adding the trainable bias `b_1`, **broadcasted** across the `inputDimensions`.
3. Apply the ReLU activation function to all elements.
4. Transform the result by multiplying with the trainable matrix `W_2` and adding the trainable bias `b_2` (also broadcasted across the `hiddenDimensions`).

> For now, we can use identity matrices for `W_1` and `W_2`, and zero vectors for `b_1` and `b_2`. Just like with `W_0` from earlier, we'll train these parameters later on.

## Resources

- [Broadcasting](https://medium.com/@weidagang/understanding-broadcasting-in-numpy-c44dceae42ea)
- [Introduction to ReLU](https://builtin.com/machine-learning/relu-activation-function)
- [ReLU Activation (Wikipedia)](<https://en.wikipedia.org/wiki/Rectifier_(neural_networks)>)
- [Original paper](http://services.ini.uzh.ch/admin/extras/doc_get.php?id=41838) (consider using [SciSpace](https://typeset.io/chat-pdf) to help read it)

## Testing

Using identity matrices for `W_1` and `W_2` and zero vectors for `b_1` and `b_2`, we're basically just testing the ReLU activation function.

```javascript
const inputMatrix = [
  [1, -2],
  [-3, 4],
];

// identity matrix
const W_1 = [
  [1, 0],
  [0, 1],
];

// zero vector
const b_1 = [0, 0];

// identity matrix
const W_2 = [
  [1, 0],
  [0, 1],
];

// zero vector
const b_2 = [0, 0];

feedForward(inputMatrix, W_1, b_1, W_2, b_2);
```

Expected output:

```javascript
[
  [1, 0],
  [0, 4],
];
```
