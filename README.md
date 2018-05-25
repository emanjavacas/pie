# pie

- Future work:
  - Add GRL regularization on domain/source labels (which seems to help POS [https://arxiv.org/pdf/1805.06093.pdf](here: Table 1&2). We can use file names, or derive appropriate labels from file names.
  - Train both linear and char-level decoders.
  - Fine-tune threshold on dev data to see when to pick which output. 
  - Implement a mixture-model to learn to decide whether to retrieve a lemma from the cache or go and generate it one character at a time [https://arxiv.org/pdf/1609.07843.pdf](similar to this paper).
  - Morphology: people seem to just run same kind of linear classifier (one per label)
  - In what sense is POS not a morphology-label like NUM, TENSE, etc...
  - Fine-grain error analyses:
	- classify lemmatas based on edit-distance with the source (different bins, show accuracy per bin)
	- does contextual information help for lemmatization (compare to a baseline where decoding is done char-level but unconditionally, compare to linear decoding baseline without contextual information)?
