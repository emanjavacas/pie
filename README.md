
# PIE: A Framework for Joint Learning of Sequence Labelling Tasks

[![DOI](https://zenodo.org/badge/131014015.svg)](https://zenodo.org/badge/latestdoi/131014015)


PIE was primarily conceived to make experimentation on sequence labelling of variation-rich languages easy and user-friendly. PIE has been tested mostly for Lemmatization but other SoTA accuracies from other tasks like POS have been reproduced (cf. Plank et al ). PIE is *highly* configurable both in terms of input preprocessing and model definition, in principle not requiring users to write any code (instead experiments are defined with json files). It is highly modular and therefore easy to extend. It includes transductive lemmatization as an additional sequence labelling task and, finally, it is reasonably fast and memory efficient.

Documentation is work in progress and it will improve over the following months, for now the best is to check `pie/default_settings.json` which explains all input parameters and shows a full example of a config file.

Model description and evaluation results are also in preparation.

In order to run PIE, the easiest way is to download the repository and install the dependencies (see requirements.txt). Training models is done with `python train.py path/to/config.json`. All non-nested parameters can be overwritten directly from the command line using environment variables like `PIE_DEVICE=cpu` (for input parameter `device`. Warning: bear in mind that due to the way bash parses environment variables `PIE_...=False` will be parsed into a boolean `True`, which might be counterintuitive. If you wish to get `False` for a parameter from the command line you can use `PIE_...=""`).


# Configuration files

Training a model only requires a model specification and paths to training and dev datasets. Pie user interface employs a simple json file (in order to allow in-line comments, we make use of the package `JSON_minify`), an example of which can be seen below:

```json
{
  "modelname": "lemmatization-latin",
  "modelpath": "models",
  "word_max_size": 50000,
  "max_sent_len": 35,
  "tasks": [
    {
      "name": "lemma",
      "target": true,
      "context": "sentence",
      "level": "char",
      "decoder": "attentional",
      "settings": {
        "bos": true,
        "eos": true,
        "lower": true,
        "target": "lemma"
      },
      "layer": -1
    }
  ],
  "batch_size": 25,
  "patience": 3,
  "dropout": 0.25,
  "lr": 0.001,
  "lr_factor": 0.75,
  "lr_patience": 2,
  "epochs": 100,
  "cell": "GRU",
  "num_layers": 1,
  "hidden_size": 150,
  "wemb_dim": 0,
  "cemb_dim": 300,
  "cemb_type": "rnn",
  "cemb_layers": 2,
  "report_freq": 200,
  "input_path": "datasets/LLCT1/train.tsv",
  "dev_path": "datasets/LLCT1/dev.tsv",
  "char_max_size": 500,
  "char_min_freq": 1,
  "word_min_freq": 1,
  "sep": "\t",
  "task_defaults": {
    "level": "token",
    "layer": -1,
    "decoder": "linear",
    "context": "sentence"
  },

  "buffer_size": 10000,
  "shuffle": true,
  "optimizer": "Adam",
  "clip_norm": 5,
}
```
