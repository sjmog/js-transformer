// test if two things are equal
// works with matrices and arrays
// but not if they contain objects
const testEquals = (name, fn, expected, successes, failures) => {
    process.stdout.write(`- ${name}...`);
  let result;
  try {
    result = fn();
  } catch (e) {
    process.stdout.write(`\x1b[31mFailed\x1b[0m\n`);
    failures.push({ name, expected, error: e })
    return;
  }

  if (result.toString() != expected.toString()) {
    process.stdout.write(`\x1b[31mFailed\x1b[0m\n`);
    failures.push({ name, expected, result });
  } else {
    process.stdout.write(`\x1b[32mPassed\x1b[0m\n`);
    successes.push({ name });
  }
};

export { testEquals };