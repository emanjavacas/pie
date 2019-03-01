
# Can be run with python -m pie.scripts.tag_pipe
import sys
from pie import utils
from pie.tagger import Tagger, simple_tokenizer


def run(model_spec, device, batch_size, lower, beam_width, use_beam, tokenize):
    with utils.shutup():
        tagger = Tagger(device=device, batch_size=batch_size, lower=lower)

        for model, tasks in model_spec:
            tagger.add_model(model, *tasks)
            tasks = tasks or tagger.models[-1][0].label_encoder.tasks

    header = False
    for line in sys.stdin:
        if not line:
            continue

        if tokenize:
            line = simple_tokenizer(line, lower)
        else:
            line = line.split()

        preds, tasks = tagger.tag(
            [line], [len(line)], use_beam=use_beam, beam_width=beam_width)

        if not header:
            print('\t'.join(['token'] + tasks))
            header = True

        preds = preds[0]  # unpack
        tokens, tags = zip(*preds)
        for token, tags in zip(tokens, tags):
            print('\t'.join([token] + list(tags)))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model_spec', type=utils.model_spec)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--use_beam', action='store_true')
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--beam_width', default=10, type=int)
    parser.add_argument('--lower', action='store_true')
    parser.add_argument('--tokenize', action='store_true')
    args = parser.parse_args()
    run(model_spec=args.model_spec, device=args.device,
        batch_size=args.batch_size,
        lower=args.lower, beam_width=args.beam_width,
        use_beam=args.use_beam, tokenize=args.tokenize)
