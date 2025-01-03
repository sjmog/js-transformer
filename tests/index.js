import tokenize from "../src/tokenize.js";
import embed from "../src/embedding.js";
import dotProductAttention from "../src/dotProductAttention.js";
import { testEquals } from "./helpers.js";

console.log("Running tests...\n");

const successes = [];
const failures = [];

testEquals(
  "tokenize",
  () => {
    return tokenize("hello world");
  },
  [1, 2],
  successes,
  failures
);

testEquals(
  "tokenize",
  () => {
    return tokenize("hello world hello sam");
  },
  [1, 2, 1, 4],
  successes,
  failures
);

testEquals(
  "tokenize",
  () => {
    return tokenize("whassup");
  },
  [0],
  successes,
  failures
);

testEquals(
  "embed",
  () => {
    const embeddingMatrix = [
      [0.01, 0.02],
      [0.03, 0.04],
    ];
    return embed([1, 2, 1], embeddingMatrix);
  },
  [
    [0.01, 0.02],
    [0.03, 0.04],
    [0.01, 0.02],
  ],
  successes,
  failures
);

testEquals(
  "dotProductAttention",
  () => {
    const Q = [
      [1, 0],
      [0, 1],
    ];
    const K = [
      [1, 2],
      [3, 4],
    ];
    const V = [
      [5, 6],
      [7, 8],
    ];
    return dotProductAttention(Q, K, V);
  },
  [
    [6.61, 7.61],
    [6.61, 7.61],
  ],
  successes,
  failures
);

failures.forEach(({ name, expected, error, result }) => {
  if (error) {
    console.error(`\nTest ${name} failed with error`);
    console.log("Expected:", expected);
    console.log("Got:");
    console.error(error);
  } else {
    console.error(`\nTest ${name} failed`);
    console.log("Expected:", expected);
    console.log("Got:", result);
  }
});

console.log(
  `\n${successes.length + failures.length} tests complete, ${
    successes.length
  } successes, ${failures.length} failures`
);
