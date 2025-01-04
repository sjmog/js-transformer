import { MODEL_DIMENSIONS } from "../src/constants.js";
import vocabulary from "../src/vocabulary.json" assert { type: "json" };
import tokenize from "../src/tokenize.js";
import embed, { defaultEmbeddingMatrix } from "../src/embedding.js";
import dotProductAttention from "../src/dotProductAttention.js";
import { runTests, test } from "./helpers.js";

runTests([
  test(
    "tokenize converts input sequences to token IDs",
    () => tokenize("hello world hello sam"),
    "equals",
    [1, 2, 1, 4]
  ),
  test(
    "tokenize converts unknown vocabulary to 0",
    () => tokenize("whassup"),
    "equals",
    [0]
  ),
  test(
    "defaultEmbeddingMatrix has correct dimensions",
    () => [defaultEmbeddingMatrix.length, defaultEmbeddingMatrix[0].length],
    "equals",
    [vocabulary.length, MODEL_DIMENSIONS]
  ),
  test(
    "defaultEmbeddingMatrix contains non-zero values",
    () =>
      defaultEmbeddingMatrix.every((vector) =>
        vector.every((value) => value !== 0)
      ),
    "equals",
    true
  ),
  test(
    "embed converts token IDs to embeddings",
    () => {
      const testEmbeddingMatrix = [
        [0.01, 0.02],
        [0.03, 0.04],
      ];
      return embed([1, 2, 1], testEmbeddingMatrix);
    },
    "equals",
    [
      [0.01, 0.02],
      [0.03, 0.04],
      [0.01, 0.02],
    ]
  ),
  test(
    "dotProductAttention returns correct attention distribution",
    () =>
      dotProductAttention(
        [
          [1, 0],
          [0, 1],
        ],
        [
          [1, 2],
          [3, 4],
        ],
        [
          [5, 6],
          [7, 8],
        ]
      ),
    "equals",
    [
      [6.61, 7.61],
      [6.61, 7.61],
    ]
  ),
  test(
    "dotProductAttention raises error if Q and K are not the same length",
    () =>
      dotProductAttention(
        [
          [1, 0],
          [0, 1],
        ],
        [
          [1, 2],
          [3, 4],
          [5, 6],
        ],
        [
          [5, 6],
          [7, 8],
          [9, 10],
        ]
      ),
    "raises",
    "Q and K must be the same length"
  ),
  test(
    "dotProductAttention raises error if Q and K length are not equal to the MODEL_DIMENSIONS",
    () =>
      dotProductAttention(
        [
          [1, 0],
          [0, 1],
          [0, 1],
        ],
        [
          [1, 2],
          [3, 4],
          [5, 6],
        ],
        [
          [5, 6],
          [7, 8],
          [9, 10],
        ]
      ),
    "raises",
    "Q and K length must match the MODEL_DIMENSIONS"
  ),
]);
