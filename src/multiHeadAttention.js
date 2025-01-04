import dotProductAttention from "./dotProductAttention.js";

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
  return concatenateHeads(headsOutput);
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
