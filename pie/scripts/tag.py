
# Can be run with python -m pie.scripts.tag
from pie import utils
from pie.tagger import Tagger


def run(model_spec, input_path, beam_width, use_beam, keep_boundaries, **kwargs):
    tagger = Tagger(**kwargs)

    for model, tasks in model_spec:
        tagger.add_model(model, *tasks)
        print(" - model: {}".format(model))
        tasks = tasks or tagger.models[-1][0].label_encoder.tasks
        print(" - tasks: {}".format(", ".join(tasks)))

    for fpath in utils.get_filenames(input_path):
        print("Tagging file [{}]...".format(fpath))
        tagger.tag_file(fpath, use_beam=use_beam, beam_width=beam_width,
                        keep_boundaries=keep_boundaries)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model_spec', type=utils.model_spec)
    parser.add_argument('input_path', help="unix string")
    parser.add_argument('--keep_boundaries', action='store_true')
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--use_beam', action='store_true')
    parser.add_argument('--beam_width', default=10, type=int)
    parser.add_argument('--lower', action='store_true')
    parser.add_argument('--max_sent_len', type=int, default=35)
    parser.add_argument('--vrt', action='store_true')
    args = parser.parse_args()

    run(args.model_spec, args.input_path,
        beam_width=args.beam_width, use_beam=args.use_beam,
        keep_boundaries=args.keep_boundaries,
        device=args.device, batch_size=args.batch_size,
        lower=args.lower, max_sent_len=args.max_sent_len,
        vrt=args.vrt)
