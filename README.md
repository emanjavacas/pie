
# PIE: A Framework for Joint Learning of Sequence Labeling Tasks

[![DOI](https://zenodo.org/badge/131014015.svg)](https://zenodo.org/badge/latestdoi/131014015)

PIE was primarily conceived to make experimentation on sequence labeling of variation-rich languages easy and user-friendly. PIE has been tested mostly for Lemmatization but other SoTA accuracies from other tasks like POS have been reproduced (cf. Plank et al ). PIE is *highly* configurable both in terms of input preprocessing and model definition, in principle not requiring users to write any code (instead experiments are defined with json files). It is highly modular and therefore easy to extend. It includes transductive lemmatization as an additional sequence labeling task and, finally, it is reasonably fast and memory efficient.

Documentation is work in progress and it will improve over the following months. A good place to learn about its functionality is to check `pie/default_settings.json` which explains all input parameters and shows a full example of a config file (minus input data).

## Installation
In order to run PIE, the easiest way is to download the repository and install the dependencies (see `requirements.txt`). There is no need to install the packaged (and in fact there is no `setup.py` file), since there aren't any required compilation steps. The only step needed to have `pie` available from any place in the file system is to add the path to `pie` to the `PYTHONPATH` environment variable. There are two ways to accomplish this:

- From your bash init file (depending on your distro and configuration this could be `.bashrc`, `.bash_profile`, `.profile`, etc...):

```bash
export PYTHONPATH="$PYTHONPATH:/path/to/pie"
```

- From your python script, using `sys`:

```python
import sys
sys.path.append('/path/to/pie')
```

Note that this is only required if you wish to use PIE programmatically from within a python script. Otherwise, you will be fine using the provided `train.py`, `evaluate.py` and `tag.py` scripts or deploying PIE as a web-app (see `webapp/`).

## Training

Training models is done with `python train.py path/to/config.json`. All non-nested parameters can be overwritten directly from the command line using environment variables like `PIE_DEVICE=cpu` (for input parameter `device`. Warning: bear in mind that due to the way bash parses environment variables `PIE_...=False` will be parsed into a boolean `True`, which might be counter-intuitive. If you wish to get `False` for a parameter from the command line you can use `PIE_...=""`).

# Model

PIE underlying model comprises a set of hierarchical feature extractors from the character-level up to the sentence-level. For each input token a sentence-level feature vector is extracted and used for the prediction of any number of target tasks (e.g. POS-tagging, lemmatization, ...)
![](./img/PIE.svg)

Prediction is accomplished with decoder modules. We provide implementations of a `linear` decoder trained to maximize the probability assigned by the model to the corpus data via a softmax function (similar to a MaxEnt classifier). A `crf` decoder, particularly suited for tasks that imply a dependency between neighboring output tags and an `attentional` decoder, suited for tasks that can be solved by generating the token-level output character by characters in a string transduction manner (e.g. lemmatization, normalization).

# Configuration files

Training a model only requires a model specification and paths to training and dev datasets. Pie user interface employs a simple json file (in order to allow in-line comments, we make use of the package `JSON_minify`), an example of which can be seen below:

```json
{
  "modelname": "lemmatization-latin",
  "modelpath": "models",
 
  // input data
  "input_path": "datasets/LLCT1/train.tsv",
  "dev_path": "datasets/LLCT1/dev.tsv",
  "sep": "\t",

  // model definition
  "tasks": [
    {
      "name": "lemma",
      "target": true,
      "context": "sentence",
      "level": "char",
      "decoder": "attentional",
      "settings":
	  {
        "bos": true,
        "eos": true,
        "lower": true,
        "target": "lemma"
      },
      "layer": -1
    }
  ],

  // training parameters
  "batch_size": 25,
  "dropout": 0.25,
  "epochs": 100,
  "optimizer": "Adam",
  "patience": 3,
  "lr": 0.001,
  "lr_factor": 0.75,
  "lr_patience": 2,

  // model hyperparameters
  "cell": "GRU",
  "num_layers": 1,
  "hidden_size": 150,
  "wemb_dim": 0,
  "cemb_dim": 300,
  "cemb_type": "rnn",
  "cemb_layers": 2
}
```

The very minimum set of options required to train a model includes `input_path` (path to files with training data), `dev_path` (path to files with development data), and `tasks`, which defines the model to be trained.

## Example task configurations

- POS tagging using a CRF

```json
"tasks": [
  {
    "name": "pos",
    "target": true,
    "decoder": "crf",
    "layer": -1 
  }
]
```

- POS tagging using a linear decoder and auxiliary tasks
```json
{
"tasks": [
  {
    "name": "pos",
    "level": "token",
    "target": true,
    "decoder": "crf",
    "layer": -1,
    "schedule": 
    {
      "patience": 3, // stop training after 3 epochs without improvement
	},
  },
  {
    "name": "case",
    "level": "token",
    "target": false,
    "decoder": "linear",
    "layer": 0,
    "schedule": 
	{
    // halve loss contribution of this task after 2 epochs without improvement
       "patience": 2,
	   "factor": 0.5
    }
  },
  {
    "name": "number",
    "level": "token",
    "target": false,
    "decoder": "linear",
    "layer": 0,
    "schedule": 
	{
    // halve loss contribution of this task after 2 epochs without improvement
      "patience": 2,
	  "factor": 0.5
    }
  }
]}
```

To avoid verbosity, the same configuration can be written in the following form:

```json
{
"tasks": [
  {
    "name": "pos",
    "level": "token",
    "target": true,
    "decoder": "crf",
    "layer": -1,
    "schedule": 
    {
      "patience": 3, // stop training after 3 epochs without improvement
	},
  {
    "name": "case",
 
  },
  {
    "name": "number",
  }
],
"task_defaults": 
  {
    "level": "token",
    "decoder": "linear",
    "layer": 0
  },
  // halve loss contribution of auxiliary task after 2 epochs without improvement
"patience": 2,
"factor": 0.5
}
```

- Transduction-based lemmatization
```json
{
"tasks": [
  {
    "name": "lemma",
    "level": "char",
    "target": true,
    "decoder": "attentional",
	"context": "sentence", // use sentence-level features to help lemmatization
    "layer": -1,
    "schedule": 
    {
      "patience": 3, // stop training after 3 epochs without improvement
	}
  }
]}
```

# Improving feature extraction with a joint Language Model loss

Pie has a built-in option to improve feature extraction by predicting neighboring words from the sentence-level feature vectors. The mechanism has been thoroughly tested for lemmatization in research currently submitted to review and it has been shown to be very effective for languages without a fixed writing standard (e.g. historical languages) and other languages with high token-lemma ambiguity. Besides, there is nothing in theory opposing the idea that it might help with other tasks such as POS-tagging, morphological analyses, etc...  The options affecting the joint LM-loss are: `include_lm` (switch on the option), `lm_shared_softmax` (whether to share parameters between forward and backward LMs, recommended value: `true`), `lm_schedule` (parameters to lower the weight assigned to the LM loss over training, once the LM loss starts overfitting it is a good idea to start reducing the loss and eventually set it to 0 to avoid affecting the learning of the target task).

# Multi-task learning

When more than one task is defined, at least and at most one task has to have the key-value pair `"target": true`, denoting that that's the task we ultimately care about. All the other tasks will be treated as auxiliary tasks with the goal of extracting better features for the target task. Note that in any case the model will still be able to predict output for auxiliary tasks, but in the spirit of multi-task learning, you will prefer to train a separate model for each of the tasks you care about, selecting each time the appropriate target task and letting all other tasks help the optimization. In the end, you have as many models as tasks you care about, each of which has been optimized for the respective task.

## More on auxiliary tasks

An auxiliary task might help learning better features that the classifier for the target task can exploit to produce better (in terms of classification performance) and more robust output (less susceptible to spurious correlations). However, training dynamics in a multi-task setting are complex (even more so than in a normal setting), since different tasks usually results in different learning curves that have to be monitored. In particular, if an auxiliary task converges before the target task, further training might lead that auxiliary task towards overfitting, thereby undoing the potential work done so far. Moreover, losses from different tasks are usually in different scales and this might have the effect that an auxiliary task with a loss on a higher scale dominates training.

In order to avoid it, the strategy chosen for PIE consists on set learning schedules for tasks (similar to early stopping) that decrease the weights given to particular tasks over time based on development performance. 

Multi-task learning consists on jointly training a model for different tasks while sharing parts of the general architecture for all tasks. This can be accomplished by either computing the loss for all tasks every batch and aggregating it before the backward pass, or optimizing in each batch for a single task randomly sampled based on a particular distribution. PIE follows the latter setting, which is known to produce better results.

Additionally, it is also important, in case of a multi-layer sentence-level feature extractor, to select at what layer a particular task can help the most (this can be controlled with the "layer" option).

Finally, multi-task learning is far from being a silver bullet and it is an empirical question whether a multi-task learning setup will yield improvements. It is recommended to first train a single model, and then try different multi-task learning configuration to see if improvements can be achieved.

# (TODO) Webapp

# (TODO) Post-correction App
