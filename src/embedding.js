// embedding.js
import vocabulary from "./vocabulary.json" assert { type: "json" };

function makeEmbeddingMatrix() {
  const vocabSize = vocabulary.length;
  const modelDim = 2;
  let embeddingMatrix = [];

  // Initialize a random embedding matrix of shape (vocabSize x modelDim).
  // embeddingMatrix[i] will be the vector for token ID = i+1
  for (let i = 0; i < vocabSize; i++) {
    let row = [];
    for (let d = 0; d < modelDim; d++) {
      row.push(Math.random() * 0.01); // small random init
    }
    embeddingMatrix.push(row);
  }
  return embeddingMatrix;
}

export default function embed(tokenIds, embeddingMatrix = makeEmbeddingMatrix()) {
  return tokenIds.map((id) => embeddingMatrix[id - 1]);
}
