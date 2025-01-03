## NOTES
- I'm not sure if I "got" the Q, K, V vectors. I have a vague idea, but I'm not sure. That's even after reading a few resources (listed below), the paper, and writing the code.
- I felt like I was "trip-falling" my way through getting the right values for Q, K, V.
- Implementing matrix multiplication felt great.
- Implementing softmax felt great.
- Implementing the actual attention felt great, though I'm not sure I understand the output. It would be better with a real worked example here – and perhaps actual weights for Q, K, V, to have the transformer "work as I go". (Doing embedding first would have helped here.)

# Challenge 3: Implementing the Dot-Product Attention Core (Dot the ‘I’ in Attention)

## Orientation Summary
The foundation of a Transformer is **self-attention**. The simplest building block is the scaled dot-product attention formula. We’ll create a function that takes in queries, keys, and values arrays and outputs the attended result.

## New Terminology
- **Scaled Dot-Product Attention** – The formula for computing attention weights, \(\text{softmax}\!\bigl(\tfrac{QK^T}{\sqrt{d_k}}\bigr)\), then applying them to \(V\).

## Success Criteria
“In this challenge, we will create dotProductAttention.js that exports a function `dotProductAttention(Q, K, V)`. Given small numeric arrays as `Q`, `K`, `V`, the function returns the attention output.”

## Learning Objectives
- Intuit the meaning of the Q, K, and V vectors.
- Implement matrix mathematics in JavaScript such as matrix multiplication (**matMul**), matrix transposition, and scaling.
- Implement softmax in pure JavaScript. 
- Implement self-attention in JavaScript.

> Use your commit messages to explain your thought process & capture your learning. For instance, [here's mine](https://github.com/sjmog/js-transformer/commit/5e288d3d16b85dbde75425418f6ef0db75ce0082).

## Scaffold Steps (with Testing)
1. Create a new file `dotProductAttention.js`.  
2. Write a helper function `matMul(A, B)` that multiplies two matrices (2D arrays).  
3. Write a helper function `softmax(vector)` that normalizes a 1D array.  
4. Implement the formula for dot-product attention:  
   - Compute \(QK^T\).  
   - Scale by \(\sqrt{d_k}\).  
   - Apply softmax to each row.
   - Multiply the result by \(V\).  
5. Export the `dotProductAttention(Q, K, V)` function.  
6. Testing:  
   - Add a test that checks the output of `dotProductAttention(Q, K, V)` against the result below:

```javascript
const Q = [
  [1, 0],
  [0, 1]
];
const K = [
  [1, 2],
  [3, 4]
];
const V = [
  [5, 6],
  [7, 8]
];

dotProductAttention(Q, K, V) === [[6.61, 7.61], [6.61, 7.61]] // true
```

## Testing
Expected Math & Output
	1.	QK^T =

\begin{bmatrix}
1 & 0 \\
0 & 1
\end{bmatrix}
\times
\begin{bmatrix}
1 & 3 \\
2 & 4
\end{bmatrix}

\begin{bmatrix}
1\cdot 1 + 0\cdot 2 & 1\cdot 3 + 0\cdot 4 \\
0\cdot 1 + 1\cdot 2 & 0\cdot 3 + 1\cdot 4
\end{bmatrix}

\begin{bmatrix}
1 & 3 \\
2 & 4
\end{bmatrix}

	2.	Scale by \frac{1}{\sqrt{2}}\approx 0.7071:

\begin{bmatrix}
0.7071 & 2.1213 \\
1.4142 & 2.8284
\end{bmatrix}

	3.	Softmax row-wise:
	•	Row 0: [0.7071, 2.1213]
	•	\exp(0.7071)\approx 2.0285,\ \exp(2.1213)\approx 8.3371
	•	Sum \approx 10.3656
	•	Softmax => [0.1956, 0.8044]
	•	Row 1: [1.4142, 2.8284]
	•	\exp(1.4142)\approx 4.1149,\ \exp(2.8284)\approx 16.9415
	•	Sum \approx 21.0564
	•	Softmax => [0.1955, 0.8045]
So attention distribution:

\begin{bmatrix}
0.1956 & 0.8044 \\
0.1955 & 0.8045
\end{bmatrix}

(approx.)
	4.	Multiply by V = \begin{bmatrix}5 & 6 \\ 7 & 8\end{bmatrix}:

\begin{bmatrix}
0.1956 & 0.8044 \\
0.1955 & 0.8045
\end{bmatrix}
\times
\begin{bmatrix}
5 & 6 \\
7 & 8
\end{bmatrix}

\begin{bmatrix}
6.6088 & 7.6088 \\
6.6090 & 7.6090
\end{bmatrix}
\approx
\begin{bmatrix}
6.61 & 7.61 \\
6.61 & 7.61
\end{bmatrix}

## Resources
- [The Illustrated Transformer - Jay Alammar (Blog)](http://jalammar.github.io/illustrated-transformer/)
- [Softmax Wikipedia](https://en.wikipedia.org/wiki/Softmax_function#:~:text=The%20softmax%20function%2C%20also%20known,used%20in%20multinomial%20logistic%20regression.)
- [What is the intuition behind the attention mechanism?](https://www.educative.io/answers/what-is-the-intuition-behind-the-dot-product-attention)

