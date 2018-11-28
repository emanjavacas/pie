
# PIE: A Framework for Joint Learning of Sequence Labelling Tasks

[![DOI](https://zenodo.org/badge/131014015.svg)](https://zenodo.org/badge/latestdoi/131014015)

PIE was primarily conceived to make experimentation on sequence labelling of variation-rich languages easy and user-friendly. PIE has been tested mostly for Lemmatization but other SoTA accuracies from other tasks like POS have been reproduced (cf. Plank et al ). PIE is *highly* configurable both in terms of input preprocessing and model definition, in principle not requiring users to write any code (instead experiments are defined with json files). It is highly modular and therefore easy to extend. It includes transductive lemmatization as an additional sequence labelling task and, finally, it is reasonably fast.

Documentation is work in progress and it will improve over the following months, for now the best is to check `pie/default_settings.json` which explains all input parameters and shows a full example of a config file.

Model description and evaluation results are also in preparation.

In order to run PIE, the easiest way is to download the repository and install the dependencies (see requirements.txt). Training models is done with `python train.py path/to/config.json`. All non-nested parameters can be overwritten directly from the command line using environment variables like `PIE_DEVICE=cpu` (for input parameter `device`. Warning: bear in mind that `PIE_...=False` will be parsed into a boolean `True`, in order to get `False` you could use `PIE_...=""`).

- Future work:
  - Add GRL regularization on domain/source labels (which seems to help POS [https://arxiv.org/pdf/1805.06093.pdf](here: Table 1&2). We can use file names, or derive appropriate labels from file names.
  - Implement a mixture-model to learn to decide whether to retrieve a lemma from the cache or go and generate it one character at a time [https://arxiv.org/pdf/1609.07843.pdf](similar to this paper).

