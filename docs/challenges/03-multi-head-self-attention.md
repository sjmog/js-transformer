# Challenge 3: Introducing Multi-Head Self-Attention (Heads Up!)

## Orientation Summary
In Transformers, we don’t just compute attention once; we compute it in parallel with different heads, then concatenate and transform. This challenge builds on the previous step by replicating the dot-product attention multiple times.

## New Terminology
* Multi-Head Attention – Splitting embeddings into multiple “heads” to learn different aspects of relationships.

Success Criteria
“In this challenge, we will create multiHeadAttention.js that exports a function multiHeadAttention(Q, K, V, numHeads, headDim) returning the multi-head output.”

## Learning Objectives
* Slice Q, K, and V into multiple heads.
* Concatenate the resulting context vectors from each head.
* Apply a linear transformation to combine heads.

## Scaffold Steps (with Testing)
1. Create multiHeadAttention.js.
2. Write logic to:
   - Split Q, K, V into numHeads sub-tensors each of size headDim.
   - Call dotProductAttention on each sub-tensor.
   - Concatenate the outputs.
   - Optionally apply a final linear projection (a simple matrix multiply).
3. Export multiHeadAttention(Q, K, V, numHeads, headDim).
	4.	Testing:
	•	Create a test-multiHeadAttention.js.
	•	Use small Q, K, V arrays again but shape them so they can be split into 2 heads.
	•	Check that the result shape is correct. (For example, if you have 2 heads each producing a (batchSize × headDim) output, the final shape should be (batchSize × (numHeads * headDim)).)

## Resources
* Multi-Head Attention (Paper Explanation)
* Split vs. Chunk in Arrays (Example Discussion)

## Example

```javascript
// multiHeadAttention.js
import { dotProductAttention } from "./dotProductAttention.js";

export function multiHeadAttention(Q, K, V, numHeads, headDim) {
  // 1. Split Q, K, V into heads
  // 2. For each head, compute dotProductAttention
  // 3. Concatenate
  // 4. Optional: apply linear transform
  // 5. Return final combined result
}
```

```javascript
// test-multiHeadAttention.js
import { multiHeadAttention } from "./multiHeadAttention.js";

const Q = [ /* shape: (batch=2) x (numHeads*headDim=4) etc. */ ];
const K = [ /* ... */ ];
const V = [ /* ... */ ];

const numHeads = 2;
const headDim = 2;

const output = multiHeadAttention(Q, K, V, numHeads, headDim);
console.log("Multi-head output:", output);
// Check shape & partial numeric correctness.
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
  [3, 4]
];
const K = [
  [1, 2],
  [3, 4]
];
const V = [
  [5, 6],
  [7, 8]
];

const numHeads = 2;
const headDim = 1; // so each head processes 1 dimension

const result = multiHeadAttention(Q, K, V, numHeads, headDim);
console.log("Multi-head output:", result);
```

### Expected Math & Output
We handle each “head” separately, then concatenate the results horizontally.
* Head 1: uses the first dimension of each row:
* Q_1 = \begin{bmatrix}1\\3\end{bmatrix}, K_1 = \begin{bmatrix}1\\3\end{bmatrix}, V_1 = \begin{bmatrix}5\\7\end{bmatrix}.
* After dot-product attention + softmax + multiply by V_1, you get approximately:

\begin{bmatrix}6.7618 \\ 6.9951\end{bmatrix}

* Head 2: uses the second dimension of each row:
* Q_2 = \begin{bmatrix}2\\4\end{bmatrix}, K_2 = \begin{bmatrix}2\\4\end{bmatrix}, V_2 = \begin{bmatrix}6\\8\end{bmatrix}.
* That yields approximately:

\begin{bmatrix}7.964 \\ 7.9993\end{bmatrix}

* Concatenate these column-wise => final shape (2×2):

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
  [6.995, 7.999]
]
```