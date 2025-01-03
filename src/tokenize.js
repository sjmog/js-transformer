import vocabulary from "./vocabulary.json" assert { type: "json" };

export default function tokenize(text) {
  // simple space-based tokenisation  
  const words = text.split(" ");

  // use 1-based indexing
  return words.map(word => vocabulary.indexOf(word) + 1);
}