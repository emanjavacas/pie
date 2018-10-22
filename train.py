
import os
import time
import logging
from datetime import datetime

from pie.settings import settings_from_file
from pie.trainer import Trainer
from pie import initialization
from pie.data import Dataset, Reader, MultiLabelEncoder
from pie.models import SimpleModel, get_pretrained_embeddings

# set seeds
import random
import numpy
import torch

now = datetime.now()
seed = now.hour * 10000 + now.minute * 100 + now.second
print("Using seed:", seed)
random.seed(seed)
numpy.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)


def get_targets(settings):
    # infix
    targets = []
    for task in settings.tasks:
        if task.get('schedule', {}).get('target'):
            targets.append(task['name'])

    if not targets and len(settings.tasks) == 1:
        targets.append(settings.tasks[0])

    return targets

def get_fname_infix(settings):
    # fname
    fname = os.path.join(settings.modelpath, settings.modelname)
    timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    targets = get_targets(settings)

    if targets:
        infix = '-'.join(['+'.join(targets), timestamp])
    else:
        infix = timestamp

    return fname, infix


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', nargs='?', default='config.json')
    args = parser.parse_args()

    settings = settings_from_file(args.config_path)

    # datasets
    reader = Reader(settings, settings.input_path)
    tasks = reader.check_tasks(expected=None)
    label_encoder = MultiLabelEncoder.from_settings(settings, tasks=tasks)
    if settings.verbose:
        print("::: Available tasks :::")
        print()
        for task in tasks:
            print("- {}".format(task))
        print()

    # fit
    start = time.time()
    if settings.verbose:
        print("::: Fitting data :::")
        print()
    ninsts = label_encoder.fit_reader(reader)
    if settings.verbose:
        print("Found {} total instances in training set in {:g} secs".format(
            ninsts, time.time() - start))
        print()
        print("::: Vocabulary :::")
        print()
        types = '{}/{}={:.2f}'.format(*label_encoder.word.get_type_stats())
        tokens = '{}/{}={:.2f}'.format(*label_encoder.word.get_token_stats())
        print("- {:<15} types={:<10} tokens={:<10}".format("word", types, tokens))
        types = '{}/{}={:.2f}'.format(*label_encoder.char.get_type_stats())
        tokens = '{}/{}={:.2f}'.format(*label_encoder.char.get_token_stats())
        print("- {:<15} types={:<10} tokens={:<10}".format("char", types, tokens))
        print()
        print("::: Target tasks :::")
        print()
        for task, le in label_encoder.tasks.items():
            print("- {:<15} target={:<6} level={:<6} vocab={:<6}"
                  .format(task, le.target, le.level, len(le)))
        print()

    trainset = Dataset(settings, reader, label_encoder)

    devset = None
    if settings.dev_path:
        devset = Dataset(settings, Reader(settings, settings.dev_path), label_encoder)
        devset = devset.get_batches()
    elif settings.dev_split > 0:
        devset = trainset.get_dev_split(ninsts, split=settings.dev_split)
        ninsts = ninsts - (len(devset) * settings.batch_size)
    else:
        logging.warning("No devset: cannot monitor/optimize training")

    testset = None
    if settings.test_path:
        testset = Dataset(settings, Reader(settings, settings.test_path), label_encoder)

    # model
    model = SimpleModel(trainset.label_encoder,
                        settings.wemb_dim, settings.cemb_dim, settings.hidden_size,
                        settings.num_layers, dropout=settings.dropout,
                        cell=settings.cell, cemb_type=settings.cemb_type,
                        custom_cemb_cell=settings.custom_cemb_cell,
                        word_dropout=settings.word_dropout,
                        lemma_context=settings.lemma_context,
                        include_lm=settings.include_lm, pos_crf=settings.pos_crf)

    # pretrain(/load pretrained) embeddings
    if model.wemb is not None:
        if settings.pretrain_embeddings:
            print("Pretraining word embeddings")
            wemb_reader = Reader(
                settings, settings.input_path, settings.dev_path, settings.test_path)
            weight = get_pretrained_embeddings(
                wemb_reader, label_encoder, size=settings.wemb_dim,
                window=5, negative=5, min_count=1)
            model.wemb.weight.data = torch.tensor(weight, dtype=torch.float32)

        elif settings.load_pretrained_embeddings:
            print("Loading pretrained embeddings")
            if not os.path.isfile(settings.load_pretrained_embeddings):
                print("Couldn't find pretrained embeddings in: {}. Skipping...".format(
                    settings.load_pretrained_embeddings))
            initialization.init_pretrained_embeddings(
                settings.load_pretrained_embeddings, label_encoder.word, model.wemb)

        if settings.freeze_embeddings:
            model.wemb.weight.requires_grad = False

    model.to(settings.device)

    print("::: Model :::")
    print()
    print(model)
    print()
    print("::: Model parameters :::")
    print()
    trainable = sum(p.nelement() for p in model.parameters() if p.requires_grad)
    total = sum(p.nelement() for p in model.parameters())
    print("{}/{} trainable/total".format(trainable, total))
    print()

    # training
    print("Starting training")
    trainer = Trainer(settings, model, trainset, ninsts)
    scores = None
    try:
        scores = trainer.train_epochs(settings.epochs, dev=devset)
    except KeyboardInterrupt:
        print("Stopping training")
    finally:
        model.eval()

    if testset is not None:
        print("Evaluating model on test set")
        for task in model.evaluate(testset.batch_generator()).values():
            task.print_summary()

    # save model
    fpath, infix = get_fname_infix(settings)
    fpath = model.save(fpath, infix=infix, settings=settings)
    print("Saved best model to: [{}]".format(fpath))

    print("Bye!")

    if scores is not None:
        with open('{}.txt'.format('-'.join(get_targets(settings))), 'a') as f:
            line = [infix, str(seed), datetime.now().strftime("%Y_%m_%d-%H_%M_%S")] + \
                   ['{}:{:.6f}'.format(task, score) for task, score in scores.items()]
            f.write('{}\n'.format('\t'.join(line)))
