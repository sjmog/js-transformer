# Challenge 4: Introducing Multi-Head Self-Attention (Heads Up!)

## Orientation Summary

In Transformers, we don’t just compute attention once, we use **multi-head attention**. We split the embedding dimension into sub-dimensions, apply separate dot-product attentions, then concatenate.

This allows the Transformer to learn different aspects of relationships.

## Success Criteria

“In this challenge, we will create multiHeadAttention.js that exports a function multiHeadAttention(Q, K, V, numHeads, headDim) returning the multi-head output.”

## Learning Objectives

- Split Q/K/V into multiple heads.
- Run attention on each head in parallel.
- Concatenate the results correctly.

## To complete this challenge, you need to:

1.	Write a function, `multiHeadAttention`, that:
  * takes as input Q, K, V, numberOfHeads, and headDimensions.
  * Checks that Q, K, V are of the same shape `[batchSize, modelDimensions]`.
  * slices the Q, K, V matrices into numberOfHeads parts (each part is headDimensions wide. For instance, if numberOfHeads = 2, a Q matrix of `[ [1, 2], [3, 4] ]` becomes `[ [ [ 1 ], [ 3 ] ], [ [ 2 ], [ 4 ] ] ]`).
  * Calls your dot-product attention on each part (the "heads output").
  * Concatenate the heads output horizontally. For instance, if the heads output is `[ [ [ 1 ], [ 3 ] ], [ [ 2 ], [ 4 ] ] ]`, the final output is `[ [ 1, 3 ], [ 2, 4 ] ]`.
  * Applies a final linear transform (matrix multiplication), `W_0`, to the output. For now, just use an [Identity Matrix](https://en.wikipedia.org/wiki/Identity_matrix) of the correct size. Later, when we train the Transformer, we'll tell it to adjust this matrix.
2. Test your function with the following known values:
  * Q = `[ [1, 2], [3, 4] ]`
  * K = `[ [1, 2], [3, 4] ]`
  * V = `[ [5, 6], [7, 8] ]`
  * numberOfHeads = 2
  * headDimensions = 1
  * The expected output is `[ [ 6.762, 7.964 ], [ 6.995, 7.999 ] ]`.

> Remember to check for rounding errors.

## Resources

- [Attention is All You Need](https://arxiv.org/abs/1706.03762)
- [Multi-Head Attention](https://www.youtube.com/watch?v=nYkdrAPrdcg)

## Walkthrough

Multi-Head Attention: Conceptual Walkthrough

1. Input Shapes

- Assume you have Q, K, V each of shape [\text{batchSize}, \text{modelDim}].
- You also have numHeads and headDim such that \text{modelDim} = \text{numHeads} \times \text{headDim}.

2. Splitting into Heads

- For each row in Q (size \text{modelDim}), you slice it into numHeads chunks, each chunk of size headDim.
- Repeat for K and V.
- Now you effectively have sub-tensors \text{Q}\text{head}, \text{K}\text{head}, \text{V}\_\text{head} of shape \text{batchSize} \times \text{headDim} for each head.

3. Dot-Product Attention per Head

- For each head, compute:

\text{scores} = Q*{\text{head}} K*{\text{head}}^T / \sqrt{\text{headDim}}
\quad\rightarrow\quad \text{softmax}(\text{scores})

- \quad\rightarrow\quad \text{softmax}(\text{scores}) \times V\_{\text{head}}
- This produces one output of shape \text{batchSize} \times \text{headDim} per head.

4. Concatenation

- Once you have all heads’ outputs (each \text{batchSize} \times \text{headDim}), you concat them horizontally (along dimension = headDim).
- The final shape: \text{batchSize} \times (\text{numHeads}\times\text{headDim}), which is again \text{batchSize}\times\text{modelDim}.

### Implementation

```js
import dotProductAttention from "./dotProductAttention.js";
import { identityMatrix, matMul } from "./helpers.js";
/**
 * multiHeadAttention
 * @param {number[][]} Q - shape (batchSize x modelDim)
 * @param {number[][]} K - same shape as Q
 * @param {number[][]} V - same shape as Q
 * @param {number} numHeads
 * @param {number} headDim
 * @returns {number[][]} - shape (batchSize x (numHeads*headDim)) i.e. (batchSize x modelDim)
 */
export default (
  Q,
  K,
  V,
  {
    // numberOfHeads is the number of attention heads
    numberOfHeads,
    // headDimensions is the dimensionality of each attention head
    headDimensions,
  }
) => {
  // The number of heads and the dimensionality of each head determine the dimensionality of the model.
  // modelDimensions = numberOfHeads * headDimensions.
  const modelDimensions = numberOfHeads * headDimensions;

  // The "batch size" is the number of rows in Q, K, and V.
  const batchSize = Q.length;

  // Check you have Q, K, V each of shape [batchSize, modelDimensions].
  if (
    Q.length !== batchSize ||
    K.length !== batchSize ||
    V.length !== batchSize ||
    Q[0].length !== modelDimensions ||
    K[0].length !== modelDimensions ||
    V[0].length !== modelDimensions
  ) {
    throw new Error(
      "Q/K/V matrix dimensions must match the numberOfHeads * headDimensions"
    );
  }

  const Q_parts = [];
  const K_parts = [];
  const V_parts = [];

  // for each head, slice Q, K, and V into numberOfHeads sub-matrices.
  // This gives you QKV sub-tensors of shape [batchSize, headDimensions] for each head.
  // e.g. if Q = [ [1, 2], [3, 4] ]
  // Q_parts = [ [ [ 1 ], [ 3 ] ], [ [ 2 ], [ 4 ] ] ]
  for (let headIndex = 0; headIndex < numberOfHeads; headIndex++) {
    const startCol = headIndex * headDimensions;
    const endCol = startCol + headDimensions;
    Q_parts.push(sliceMatrix(Q, startCol, endCol));
    K_parts.push(sliceMatrix(K, startCol, endCol));
    V_parts.push(sliceMatrix(V, startCol, endCol));
  }

  const headsOutput = [];

  // For each head, you can now apply dotProductAttention.
  // This produces one output of shape [batchSize, headDimensions] per head.
  for (let headIndex = 0; headIndex < numberOfHeads; headIndex++) {
    const attentionPart = dotProductAttention(
      Q_parts[headIndex],
      K_parts[headIndex],
      V_parts[headIndex]
    );
    headsOutput.push(attentionPart);
  }

  // Now you concat all the heads' outputs horizontally (along dimension = headDim).
  // The final shape: batchSize * (numHeads * headDim), which is back to batchSize * modelDim.
  const concatenatedHeads = concatenateHeads(headsOutput);

  // Apply a final linear transform, `W_0`, to the output.
  // For now, just use an identity matrix of the correct size.
  // Later, when we train the Transformer, we'll tell it to adjust this matrix.
  const W_0 = identityMatrix(modelDimensions);
  return matMul(concatenatedHeads, W_0);
};

/**
 * sliceColumns: returns sub-matrix of 'matrix' from [startCol..endCol).
 * @param {number[][]} matrix - shape (batchSize x totalCols)
 * @param {number} startCol
 * @param {number} endCol
 * @returns {number[][]} - shape (batchSize x (endCol - startCol))
 */
const sliceMatrix = (matrix, startCol, endCol) => {
  return matrix.map((row) => row.slice(startCol, endCol));
};

/**
 * concatenateHeads: merges an array of matrices horizontally.
 * E.g. headsOutput = [
 *   (batchSize x headDim),
 *   (batchSize x headDim),
 *   ...
 * ]
 * => final shape (batchSize x (numHeads*headDim))
 * @param {number[][][]} headsOutput - shape (numHeads x batchSize x headDim)
 * @returns {number[][]} - shape (batchSize x (numHeads*headDim))
 */
const concatenateHeads = (headsOutput) => {
  const batchSize = headsOutput[0].length;
  const numHeads = headsOutput.length;

  let result = [];
  for (let i = 0; i < batchSize; i++) {
    let newRow = [];
    for (let headIndex = 0; headIndex < numHeads; headIndex++) {
      newRow = newRow.concat(headsOutput[headIndex][i]);
    }
    result.push(newRow);
  }
  return result;
};
```

## TESTING

Below is a tiny multi-head example with 2 heads (numHeads=2, headDim=1).
We’ll not apply a final linear transform in this test so you can see the raw multi-head concatenation.

```javascript
// test-multiHeadAttention.js
import { multiHeadAttention } from "./multiHeadAttention.js";

// We'll define Q,K,V each as shape (2 x 2).
// "modelDim"=2 => "numHeads"=2 => "headDim"=1 for each head.

// Q = [ [1,2],
//       [3,4] ]
// K = [ [1,2],
//       [3,4] ]
// V = [ [5,6],
//       [7,8] ]
// We'll assume your multiHeadAttention splits each row's 2 elements
// into 2 heads, each of dimension 1.

const Q = [
  [1, 2],
  [3, 4],
];
const K = [
  [1, 2],
  [3, 4],
];
const V = [
  [5, 6],
  [7, 8],
];

const numHeads = 2;
const headDim = 1; // so each head processes 1 dimension

const result = multiHeadAttention(Q, K, V, numHeads, headDim);
console.log("Multi-head output:", result);
```

### Expected Math & Output

We handle each “head” separately, then concatenate the results horizontally.

- Head 1: uses the first dimension of each row:
- Q_1 = \begin{bmatrix}1\\3\end{bmatrix}, K_1 = \begin{bmatrix}1\\3\end{bmatrix}, V_1 = \begin{bmatrix}5\\7\end{bmatrix}.
- After dot-product attention + softmax + multiply by V_1, you get approximately:

\begin{bmatrix}6.7618 \\ 6.9951\end{bmatrix}

- Head 2: uses the second dimension of each row:
- Q_2 = \begin{bmatrix}2\\4\end{bmatrix}, K_2 = \begin{bmatrix}2\\4\end{bmatrix}, V_2 = \begin{bmatrix}6\\8\end{bmatrix}.
- That yields approximately:

\begin{bmatrix}7.964 \\ 7.9993\end{bmatrix}

- Concatenate these column-wise => final shape (2×2):

\begin{bmatrix}
6.7618 & 7.964 \\
6.9951 & 7.9993
\end{bmatrix}
\approx
\begin{bmatrix}
6.762 & 7.964 \\
6.995 & 7.999
\end{bmatrix}

Hence your console output should be close to:

```javascript
[
  [6.762, 7.964],
  [6.995, 7.999],
];
```

> Remember to check for rounding errors.
