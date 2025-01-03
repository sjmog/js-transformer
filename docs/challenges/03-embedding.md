## NOTES

* Is it true that the model will learn to embed tokens more accurately? Our embedding matrix is initialized randomly – perhaps we'll be training it later on somehow.

# Challenge 3: Embedding

## Title & Introduction

Just like how Transformer models don't work directly with input strings, they also don't work directly with arrays of token IDs. They work with **vector representations** of tokens:

``` js
// [0.1, 0.2] is a vector representation of the token "hello"
transformer([[0.1, 0.2]]) => "world"
```

### What is a vector representation of a token?

A vector representation of a token is a way of representing a token in a "vector space", where more closely-related tokens can be grouped together.

The size of the vector space is called the **model dimension**. A higher model dimension means the model is "smarter" at representing the relationship between tokens.

For example, imagine these points in a 3D space (a model dimension of 3):

- The word "King" might be represented as [0.1, 0.2, 0.3].
- The word "Queen" might be represented as [0.2, 0.3, 0.4].
- The word "Apple" might be represented as [0.8, 0.9, 0.9].

Notice how "King" and "Queen" are closer to each other than "King" and "Apple". That's because they're more closely related.

> We sometimes call this "vector space" the "latent space" because it's a hidden space where the model learns to represent tokens in a psuedo-conceptual way.

### How do we make vector representations of tokens?

The process of translating a token into a vector is called **embedding**.

> Before a Transformer is trained, the embedding layer is randomly initialized. Later on, we'll train the model, and it will learn to embed tokens more accurately.

## Learning Objectives
* Implement a trainable embedding that maps token IDs to numeric vectors.

## Scaffold
1. Initialise an `embeddingMatrix` of size `vocabSize` x `modelDim`. Use a model dimension of 2 for now. This matrix should be full of randomly-initialized vectors (arrays containing randomly-initialized floats).
2. Create a function `embed` that takes a sequence of token IDs and returns the embeddings for that sequence.

## Resources
* [A beginner's guide to tokens, vectors, and embeddings](https://medium.com/@saschametzger/what-are-tokens-vectors-and-embeddings-how-do-you-create-them-e2a3e698e037)

## Walkthrough (Detailed Checks & Code Snippets)

```js
// embedding.js
import vocabulary from "./vocabulary.json";

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

export function embed(tokenIds) {
  // For each token ID, pick the row from embeddingMatrix
  return tokenIds.map(id => embeddingMatrix[id - 1]);
}
```