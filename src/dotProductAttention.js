import { MODEL_DIMENSIONS } from "./constants.js";
import {
  matMul,
  matTranspose,
  matScale,
  matRound,
  softmax,
} from "./helpers.js";

export default (
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
  // this must match the number of dimensions in the Query and Key vectors
  if (Q.length !== K.length) {
    throw new Error("Q and K must be the same length");
  }

  if (Q.length !== MODEL_DIMENSIONS) {
    throw new Error("Q and K length must match the MODEL_DIMENSIONS");
  }

  const d_k = MODEL_DIMENSIONS;

  // first, we calculate the dot product of Q and K, which
  // requires us to transpose K to get the correct dimensions.
  // This is QK^T.
  const QK_T = matMul(Q, matTranspose(K));

  // next, we scale QK^T by the square root of the model dimensionality
  // to prevent the dot product from becoming too large.
  // This is QK^T / sqrt(d_k).
  const scaledQK_T = matScale(QK_T, 1 / Math.sqrt(d_k));

  // finally, we apply the softmax function to the scaled dot product
  // to get the attention distribution.
  // This is the softmax(QK^T / sqrt(d_k)) for each row of scaledQK_T.
  const attentionDistribution = scaledQK_T.map((row) => softmax(row));

  // we then multiply the attention distribution by V to get the output
  // This is V * softmax(QK^T / sqrt(d_k)).
  return matRound(matMul(attentionDistribution, V), 2);
};
