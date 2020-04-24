
# Can be run with python -m pie.scripts.train
import time
import os
from datetime import datetime
import logging

import pie
from pie.settings import settings_from_file
from pie.trainer import Trainer
from pie import initialization
from pie.data import Dataset, Reader, MultiLabelEncoder
from pie.models import SimpleModel, get_pretrained_embeddings
from pie import optimize

# set seeds
import random
import numpy
import torch


def get_targets(settings):
    return [task['name'] for task in settings.tasks if task.get('target')]


def get_fname_infix(settings):
    # fname
    fname = os.path.join(settings.modelpath, settings.modelname)
    timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    infix = '+'.join(get_targets(settings)) + '-' + timestamp
    return fname, infix


def run(settings):
    now = datetime.now()

    # set seed
    seed = now.hour * 10000 + now.minute * 100 + now.second
    print("Using seed:", seed)
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    if settings.verbose:
        logging.basicConfig(level=logging.INFO)

    # datasets
    reader = Reader(settings, settings.input_path)
    tasks = reader.check_tasks(expected=None)
    if settings.verbose:
        print("::: Available tasks :::")
        print()
        for task in tasks:
            print("- {}".format(task))
        print()

    # label encoder
    label_encoder = MultiLabelEncoder.from_settings(settings, tasks=tasks)
    if settings.verbose:
        print("::: Fitting data :::")
        print()
    label_encoder.fit_reader(reader)

    if settings.verbose:
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
        print("::: Tasks :::")
        print()
        for task, le in label_encoder.tasks.items():
            print("- {:<15} target={:<6} level={:<6} vocab={:<6}"
                  .format(task, le.target, le.level, len(le)))
        print()

    trainset = Dataset(settings, reader, label_encoder)

    devset = None
    if settings.dev_path:
        devset = Dataset(settings, Reader(settings, settings.dev_path), label_encoder)
    else:
        logging.warning("No devset: cannot monitor/optimize training")

    # model
    model = SimpleModel(
        label_encoder, settings.tasks,
        settings.wemb_dim, settings.cemb_dim, settings.hidden_size,
        settings.num_layers, cell=settings.cell,
        # dropout
        dropout=settings.dropout, word_dropout=settings.word_dropout,
        # word embeddings
        merge_type=settings.merge_type, cemb_type=settings.cemb_type,
        cemb_layers=settings.cemb_layers, custom_cemb_cell=settings.custom_cemb_cell,
        # lm joint loss
        include_lm=settings.include_lm, lm_shared_softmax=settings.lm_shared_softmax,
        # decoder
        scorer=settings.scorer, linear_layers=settings.linear_layers)

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
                print("Couldn't find pretrained eembeddings in: {}".format(
                    settings.load_pretrained_embeddings))
            initialization.init_pretrained_embeddings(
                settings.load_pretrained_embeddings, label_encoder.word, model.wemb)

    # load pretrained weights
    if settings.load_pretrained_encoder:
        model.init_from_encoder(pie.Encoder.load(settings.load_pretrained_encoder))

    # freeze embeddings
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

    running_time = time.time()
    trainer = Trainer(settings, model, trainset, reader.get_nsents())
    scores = None
    try:
        scores = trainer.train_epochs(settings.epochs, devset=devset)
    except KeyboardInterrupt:
        print("Stopping training")
    finally:
        model.eval()
    running_time = time.time() - running_time

    if settings.test_path:
        print("Evaluating model on test set")
        testset = Dataset(settings, Reader(settings, settings.test_path), label_encoder)
        for task in model.evaluate(testset, trainset).values():
            task.print_summary()

    # save model
    fpath, infix = get_fname_infix(settings)
    if not settings.run_test:
        fpath = model.save(fpath, infix=infix, settings=settings)
        print("Saved best model to: [{}]".format(fpath))

    if devset is not None and not settings.run_test:
        scorers = model.evaluate(devset, trainset)
        scores = []
        for task in sorted(scorers):
            scorer = scorers[task]
            result = scorer.get_scores()
            for acc in result:
                scores.append('{}-{}:{:.6f}'.format(
                    acc, task, result[acc]['accuracy']))
                scores.append('{}-{}-support:{}'.format(
                    acc, task, result[acc]['support']))
        path = '{}.results.{}.csv'.format(
            settings.modelname, '-'.join(get_targets(settings)))
        with open(path, 'a') as f:
            line = [infix, str(seed), str(running_time)]
            line += scores
            f.write('{}\n'.format('\t'.join(line)))

    print("Bye!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', nargs='?', default='config.json')
    parser.add_argument('--opt_path', help='Path to optimization file (see opt.json)')
    parser.add_argument('--n_iter', type=int, default=20)
    args = parser.parse_args()

    settings = settings_from_file(args.config_path)

    if args.opt_path:
        opt = optimize.read_opt(args.opt_path)
        optimize.run_optimize(run, settings, opt, args.n_iter)
    else:
        run(settings)
