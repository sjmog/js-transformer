export const runTests = (tests) => {
  console.log("Running tests...\n");
  const successes = [];
  const failures = [];

  tests.forEach((test) => {
    test(successes, failures);
  });

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
};

export const test = (name, fn, type, expected) => (successes, failures) => {
  const stdout = process.stdout;

  stdout.write(`- ${name}...`);
  let result;

  if (type === "raises") {
    try {
      result = fn();
      stdout.write(`\x1b[31mFailed\x1b[0m\n`);
      failures.push({ name, expected, result: "No error raised" });
      return;
    } catch (e) {
      if (e.message === expected) {
        stdout.write(`\x1b[32mPassed\x1b[0m\n`);
        successes.push({ name });
      } else {
        stdout.write(`\x1b[31mFailed\x1b[0m\n`);
        failures.push({ name, expected, error: e });
        return;
      }
    }
  }

  // test if two things are equal
  // works with matrices and arrays
  // but not if they contain objects
  if (type === "equals") {
    try {
      result = fn();
    } catch (e) {
      stdout.write(`\x1b[31mFailed\x1b[0m\n`);
      failures.push({ name, expected, error: e });
      return;
    }

    if (result.toString() != expected.toString()) {
      stdout.write(`\x1b[31mFailed\x1b[0m\n`);
      failures.push({ name, expected, result });
    } else {
      stdout.write(`\x1b[32mPassed\x1b[0m\n`);
      successes.push({ name });
    }
  }
};
