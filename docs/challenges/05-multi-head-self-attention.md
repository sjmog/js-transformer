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
  * Applies a final linear transform, `W_0`, to the output. For now, just use an [Identity Matrix](https://en.wikipedia.org/wiki/Identity_matrix) of the correct size. Later, when we train the Transformer, we'll tell it to adjust this matrix.
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
// multiHeadAttention.js
import { dotProductAttention } from "./dotProductAttention.js";

/**
 * multiHeadAttention
 * @param {number[][]} Q - shape (batchSize x modelDim)
 * @param {number[][]} K - same shape as Q
 * @param {number[][]} V - same shape as Q
 * @param {number} numHeads
 * @param {number} headDim
 * @returns {number[][]} - shape (batchSize x (numHeads*headDim)) i.e. (batchSize x modelDim)
 */
export function multiHeadAttention(Q, K, V, numHeads, headDim) {
  const batchSize = Q.length;
  const modelDim = numHeads * headDim;

  // Sanity checks
  if (
    Q[0].length !== modelDim ||
    K[0].length !== modelDim ||
    V[0].length !== modelDim
  ) {
    throw new Error("Q/K/V dimensions must match numHeads*headDim.");
  }

  // We'll store each head's output in an array
  const headsOutput = [];

  // For each head:
  for (let h = 0; h < numHeads; h++) {
    // Slice Q, K, V columns for head h
    // For example, if headDim=2, h=0 => columns 0..1, h=1 => columns 2..3, etc.
    const startCol = h * headDim;
    const endCol = startCol + headDim;

    // Build sub-matrices Q_h, K_h, V_h of shape (batchSize x headDim)
    let Q_h = sliceColumns(Q, startCol, endCol);
    let K_h = sliceColumns(K, startCol, endCol);
    let V_h = sliceColumns(V, startCol, endCol);

    // Dot-product attention on these sub-matrices
    // This returns shape (batchSize x headDim)
    let attnOut = dotProductAttention(Q_h, K_h, V_h);

    headsOutput.push(attnOut);
  }

  // Now we need to concatenate all heads horizontally
  // e.g. each head is (batchSize x headDim) => final => (batchSize x (numHeads*headDim))
  const concatenated = concatenateHeads(headsOutput); // implement a helper function

  // Optionally, you could apply a final linear projection here:
  // let final = matMul(concatenated, W_o) + b_o
  // We'll skip that in this example.

  return concatenated;
}

/**
 * sliceColumns: returns sub-matrix of 'matrix' from [startCol..endCol).
 * @param {number[][]} matrix - shape (batchSize x totalCols)
 * @param {number} startCol
 * @param {number} endCol
 */
function sliceColumns(matrix, startCol, endCol) {
  // e.g. if matrix = [ [1,2,3,4], [5,6,7,8] ] and startCol=1, endCol=3
  // we return [ [2,3], [6,7] ]
  return matrix.map((row) => row.slice(startCol, endCol));
}

/**
 * concatenateHeads: merges an array of matrices horizontally.
 * E.g. headsOutput = [
 *   (batchSize x headDim),
 *   (batchSize x headDim),
 *   ...
 * ]
 * => final shape (batchSize x (numHeads*headDim))
 */
function concatenateHeads(headsOutput) {
  // headsOutput might look like: [h1, h2, h3], each hX is shape (batchSize x headDim).
  // We'll assume all have same batchSize
  const batchSize = headsOutput[0].length;
  const numHeads = headsOutput.length;
  const headDim = headsOutput[0][0].length;

  // Build final array
  let result = [];
  for (let i = 0; i < batchSize; i++) {
    // row i from each head, concatenated
    let newRow = [];
    for (let h = 0; h < numHeads; h++) {
      newRow = newRow.concat(headsOutput[h][i]);
    }
    result.push(newRow);
  }
  return result;
}
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
