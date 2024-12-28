import dotProductAttention from "../dotProductAttention.js";
import { testEquals } from "./helpers.js";

console.log("Running tests...\n");

const successes = [];
const failures = [];

testEquals("smoke test", () => {
    return true;
}, true, successes, failures);

failures.forEach(({ name, expected, error, result }) => {
    if(error) {
        console.error(`\nTest ${name} failed with error`);
        console.log("Expected:", expected);
        console.log("Got:");
        console.error(error);
    } else {
        console.error(`\nTest ${name} failed`);
        console.log("Expected:", expected);
        console.log("Got:", result);
    }
})

console.log(`\n${successes.length + failures.length} tests complete, ${successes.length} successes, ${failures.length} failures`);