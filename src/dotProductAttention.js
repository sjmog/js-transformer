// transpose a matrix
const matTranspose = (matrix) => {
  return matrix[0].map((_, colIndex) => matrix.map((row) => row[colIndex]));
};

// multiply two matrices A and B
const matMul = (A, B) => {
  const rowsA = A.length;
  const colsA = A[0].length;
  const rowsB = B.length;
  const colsB = B[0].length;

  if (colsA !== rowsB) {
    throw new Error("Number of columns in A must match number of rows in B");
  }

  const result = new Array(rowsA).fill(0).map(() => new Array(colsB).fill(0));

  for (let i = 0; i < rowsA; i++) {
    for (let j = 0; j < colsB; j++) {
      for (let k = 0; k < colsA; k++) {
        result[i][j] += A[i][k] * B[k][j];
      }
    }
  }

  return result;
};

// round a matrix to a certain number of decimal places
const matRound = (matrix, places) => {
  return matrix.map((row) => row.map((x) => +x.toFixed(places)));
};

// scale a matrix by a scalar (multiply each element by the scalar)
const scale = (matrix, scalar) => {
  return matrix.map((row) => row.map((x) => x * scalar));
};

// converts a vector of K real numbers into a probability distribution of K possible outcomes
const softmax = (vector) => {
  const max = Math.max(...vector);
  const exp = vector.map((x) => Math.exp(x - max));
  const sum = exp.reduce((a, b) => a + b, 0);

  return exp.map((x) => x / sum);
};

const dotProductAttention = (
  // Q is the Query vector, which is derived from the
  // current position that the attention mechanism is focused on.
  Q,
  // K is the Key vector, which is derived from all the positions
  // in the input sequence.
  K,
  // V is the Value vector, which contains information associated
  // with each position in the input sequence
  V
) => {
  // d_k is the dimensionality of the model
  // this is the number of dimensions in the Query and Key vectors
  const d_k = Q[0].length;

  // first, we calculate the dot product of Q and K, which
  // requires us to transpose K to get the correct dimensions.
  // This is QK^T.
  const QK_T = matMul(Q, matTranspose(K));

  // next, we scale QK^T by the square root of the model dimensionality
  // to prevent the dot product from becoming too large.
  // This is QK^T / sqrt(d_k).
  const scaledQK_T = scale(QK_T, (1 / Math.sqrt(d_k)));

  // finally, we apply the softmax function to the scaled dot product
  // to get the attention distribution.
  // This is the softmax(QK^T / sqrt(d_k)) for each row of scaledQK_T.
  const attentionDistribution = scaledQK_T.map((row) => softmax(row));

  // we then multiply the attention distribution by V to get the output
  // This is V * softmax(QK^T / sqrt(d_k)).
  return matRound(matMul(attentionDistribution, V), 2);
};

export default dotProductAttention;