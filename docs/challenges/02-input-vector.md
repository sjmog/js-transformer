# Challenge 2: Creating a Tiny Vocabulary & Tokenizer

## Title & Introduction

Transformers have a **vocabulary** which determines the set of strings that can be used in an **input sequence** or output from the model.

When you feed an input sequence (a string of words) into a Transformer, the output sequence will be an array of probabilities: one for each word in the vocabulary.

```
// for a vocabulary of length n
transformer(input) -> [prob(vocabulary[0]), prob(vocabulary[1]), ... , prob(vocabulary[n])]
```

The Transformer then picks the vocabulary string with the highest probability, and outputs it.

```
transformer("hello") => "world"
```

In this challenge, we will create a small vocabulary of words that we want our NLP model to be able to work with.

Because Transformers can only process numbers, we need a way to convert both our input string into numbers. This is done through a process called **tokenization**, which splits text into integer IDs.

So secondly, you’ll create a simple tokenization scheme that splits text into integer IDs.

## Learning Objectives
* Create a vocabulary.
* Implement a space-separated tokenizer to map input strings → IDs.

## Scaffold
1. Create a JSON file, `vocabulary.json`, holding three or four words.
2. Write a function, `tokenize`, that:
  - splits an input string into words
  - assigns each unique word an integer ID based on its index in the vocabulary. For instance, if the vocabulary is `["hello", "world", "from", "transformers"]`, then `tokenize("hello world")` should return `[1, 2]`.
  - if a word is not in the vocabulary, it should be assigned an ID of `0`.
  
## Resources
* [Basics of Tokenization (Blog)](https://huggingface.co/docs/transformers/tokenizer_summary)

## Walkthrough (Detailed Checks & Code Snippets)

```js
// vocabulary.json
[
  "hello",
  "world",
  "from",
  "sam"
]
```

```js
// tokenize.js
import vocabulary from "./vocabulary.json";

function tokenize(text) {
  // simple space-based tokenisation  
  const words = text.split(" ");

  // use 1-based indexing
  return words.map(word => vocabulary.indexOf(word) + 1);
}
```

## Numeric Checks

- If your input string is `"hello world"`, it should tokenize to `[1, 2]`.
- If your input string is `"hello world from sam"`, it should tokenize to `[1, 2, 3, 4]`.
- If your input string is `"whassup"`, it should tokenize to `[0]`.
