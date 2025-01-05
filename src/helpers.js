// transpose a matrix
export const matTranspose = (matrix) => {
  return matrix[0].map((_, colIndex) => matrix.map((row) => row[colIndex]));
};

// multiply two matrices A and B
export const matMul = (A, B) => {
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
export const matRound = (matrix, places) => {
  return matrix.map((row) => row.map((x) => +x.toFixed(places)));
};

// scale a matrix by a scalar (multiply each element by the scalar)
export const matScale = (matrix, scalar) => {
  return matrix.map((row) => row.map((x) => x * scalar));
};

// converts a vector of K real numbers into a probability distribution of K possible outcomes
export const softmax = (vector) => {
  const max = Math.max(...vector);
  const exp = vector.map((x) => Math.exp(x - max));
  const sum = exp.reduce((a, b) => a + b, 0);

  return exp.map((x) => x / sum);
};

// create an identity matrix of size n
export const identityMatrix = (n) => {
  return Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));
};

// apply the ReLU activation function to a matrix
export const relu = (matrix) => {
  return matrix.map((row) => row.map((x) => Math.max(0, x)));
};

// "stretch" a 1D array to match the dimensions of a matrix
export const broadcast = (array, matrix) => {
  return matrix.map((row) => row.map((_, i) => array[i]));
};

export const matAdd = (A, B) => {
  return A.map((row, i) => row.map((value, j) => value + B[i][j]));
};
