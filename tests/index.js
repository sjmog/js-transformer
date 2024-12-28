import dotProductAttention from "../src/dotProductAttention.js";
import { testEquals } from "./helpers.js";

console.log("Running tests...\n");

const successes = [];
const failures = [];

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
