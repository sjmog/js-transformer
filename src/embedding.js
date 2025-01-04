// embedding.js
import { MODEL_DIMENSIONS } from "./constants.js";
import vocabulary from "./vocabulary.json" assert { type: "json" };

export const defaultEmbeddingMatrix = Array.from({ length: vocabulary.length }, () =>
  Array.from({ length: MODEL_DIMENSIONS }, () => Math.random() * 0.01)
);

export default (tokenIds, embeddingMatrix = defaultEmbeddingMatrix) =>
  tokenIds.map((id) => embeddingMatrix[id - 1]);
