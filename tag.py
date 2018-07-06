
import os
import re

from pie import utils
from pie.tagger import Tagger


def model_spec(inp):
    """
    >>> example = 'model-pos-2018:03:05.tar'
    >>> model_spec(example)
    [('model-pos-2018:03:05.tar', [])]

    >>> example = '<model-pos-2018:03:05.tar,pos><model-pos-2018:03:05.tar,lemma>'
    >>> model_spec(example)
    [('model-pos-2018:03:05.tar', ['pos']), ('model-pos-2018:03:05.tar', ['lemma'])]
    """
    if not inp.startswith('<'):
        return [(inp, [])]

    output = []
    for string in re.findall(r'<([^>]+)>', inp):
        model_path, *tasks = string.split(',')
        output.append((model_path, tasks))

    return output


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model_spec', type=model_spec)
    parser.add_argument('input_path', help="unix string")
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--lower', action='store_true')
    args = parser.parse_args()

    tagger = Tagger(device=args.device, batch_size=args.batch_size, lower=args.lower)

    for model, tasks in args.model_spec:
        tagger.add_model(model, *tasks)
        print(" - model: {}".format(model))
        tasks = tasks or tagger.models[-1][0].label_encoder.tasks
        print(" - tasks: {}".format(", ".join(tasks)))

    for fpath in utils.get_filenames(args.input_path):
        print("Tagging file [{}]...".format(fpath))
        tagger.tag_file(fpath)
