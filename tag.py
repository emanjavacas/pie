
import os

from pie import utils
from pie.tagger import Tagger


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model_spec', type=model_spec)
    parser.add_argument('input_path', help="unix string")
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--use_beam', action='store_true')
    parser.add_argument('--beam_width', default=10, type=int)
    parser.add_argument('--lower', action='store_true')
    args = parser.parse_args()

    tagger = Tagger(device=args.device, batch_size=args.batch_size, lower=args.lower)

    for model, tasks in utils.model_spec(args.model_spec):
        tagger.add_model(model, *tasks)
        print(" - model: {}".format(model))
        tasks = tasks or list(model.label_encoder.tasks)
        print(" - tasks: {}".format(", ".join(tasks)))

    for fpath in utils.get_filenames(args.input_path):
        print("Tagging file [{}]...".format(fpath))
        tagger.tag_file(fpath, use_beam=args.use_beam, beam_width=args.beam_width)
