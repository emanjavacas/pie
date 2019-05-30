
import json

import tqdm

from pie import utils
from pie.data import Reader, MultiLabelEncoder, Dataset
from pie.models import SimpleModel
from pie.trainer import Trainer
from pie.tagger import Tagger
from pie.settings import settings_from_file


def tag_reader(tagger, reader, batch_size=50, **kwargs):
    """
    Use a model to overwrite a particular set of tasks of an input train file
    with the output of a models trained for those particular tasks
    """
    sorted_tasks = sorted(reader.check_tasks())
    total = reader.get_nsents() // batch_size
    for sents in tqdm.tqdm(utils.chunks(reader.readsents(), batch_size), total=total):
        _, sents = zip(*sents)  # discard sentence metadata
        (sents, tasks) = zip(*sents)  # unwrap sentence data
        tag, ttasks = tagger.tag(sents, **kwargs)
        for sent, sent_tasks, tag_sent in zip(sents, tasks, tag):
            assert len(sent) == len(tag_sent)
            for idx, (tok, tag_tok) in enumerate(zip(sent, tag_sent)):
                tag_tok, tag_tasks = tag_tok
                assert tag_tok == tok
                tag_tasks = dict(zip(ttasks, tag_tasks))
                output = []
                for t in sorted_tasks:
                    output.append(tag_tasks[t] if t in tag_tasks else sent_tasks[t][idx])
                yield tok, output
            yield None


def run(settings, jackknife_n=5, serialize=True):
    # datasets
    reader = Reader(settings, settings.input_path)
    # label encoder
    label_encoder = MultiLabelEncoder.from_settings(settings).fit_reader(reader)
    if settings.verbose:
        label_encoder.summary()
    devreader = Reader(settings, settings.dev_path)
    devset = Dataset(settings, devreader, label_encoder)

    fpath, infix = settings.get_fname_infix()

    with open(utils.ensure_ext(fpath, 'jackknife.json', infix), 'w') as logf, \
            open(utils.ensure_ext(fpath, 'jackknife.tab', infix), 'w') as outf:

        # write target header
        outf.write('\t'.join(['token'] + sorted(reader.check_tasks())) + '\n')

        for split, (train, test) in enumerate(reader.jackknife(jackknife_n)):
            print("::: Starting split {}/{} :::\n".format(split + 1, jackknife_n))
            # model
            model = SimpleModel(
                label_encoder, settings.tasks,
                settings.wemb_dim, settings.cemb_dim, settings.hidden_size,
                settings.num_layers, dropout=settings.dropout,
                cell=settings.cell, cemb_type=settings.cemb_type,
                cemb_layers=settings.cemb_layers,
                custom_cemb_cell=settings.custom_cemb_cell,
                linear_layers=settings.linear_layers,
                scorer=settings.scorer,
                word_dropout=settings.word_dropout,
                lm_shared_softmax=settings.lm_shared_softmax,
                include_lm=settings.include_lm)
            model.to(settings.device)

            trainset = Dataset(settings, train, label_encoder)
            trainer = Trainer(settings, model, trainset, train.get_nsents())
            scores = trainer.train_epochs(settings.epochs, devset=devset)

            # store split performance
            logf.write(json.dumps({'scores': scores, 'split': split + 1}) + '\n')

            # tag the split
            print("::: Tagging split {}/{} :::\n".format(split + 1, jackknife_n))
            for line in tag_reader(
                    Tagger().add_model(model), test, batch_size=settings.batch_size):
                if line is not None:
                    tok, output_tasks = line
                    outf.write('\t'.join([tok, *output_tasks]) + '\n')
                else:
                    outf.write('\n')

            # (maybe) store and free up mem
            model.to('cpu')
            if serialize:
                model.save(fpath, infix + '-jackknife-{}'.format(split + 1), settings)

    # train on full
    model.to(settings.device)
    trainset = Dataset(settings, reader, label_encoder)
    scores = Trainer(settings, model, trainset, reader.get_nsents()).train_epochs(
        settings.epochs, devset=devset)
    model.save(fpath, infix, settings)
    print("Saved best model to: [{}]".format(fpath))

    # tag devset
    print("Tagging devset")
    with open(utils.ensure_ext(fpath, 'tab', 'jackknife-dev'), 'w') as outf:
        for line in tag_reader(
                Tagger().add_model(model), devreader, batch_size=settings.batch_size):
            if line is not None:
                tok, output_tasks = line
                outf.write('\t'.join([tok, *output_tasks]) + '\n')
            else:
                outf.write('\n')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path')
    parser.add_argument('--jackknife_n', type=int, default=5)
    args = parser.parse_args()
    run(settings_from_file(args.config_path), jackknife_n=args.jackknife_n)
