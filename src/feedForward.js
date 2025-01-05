import { broadcast, matMul, matAdd, relu } from "./helpers.js";

export default (inputMatrix, W_1, b_1, W_2, b_2) => {
  const broadcastedB_1 = broadcast(b_1, inputMatrix)
  const broadcastedB_2 = broadcast(b_2, inputMatrix)

  const linear1 = matAdd(matMul(inputMatrix, W_1), broadcastedB_1);
  const relu1 = relu(linear1);
  const linear2 = matAdd(matMul(relu1, W_2), broadcastedB_2);
  return linear2;
};
