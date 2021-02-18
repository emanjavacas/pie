
import os
from datetime import datetime
import logging
import random

import numpy
import torch
from transformers import AutoModel, AutoTokenizer

from pie.settings import settings_from_file, get_targets, get_fname_infix
from pie.trainer import Trainer
from pie import optimize
from pie.data import Reader, MultiLabelEncoder
from pie.models import TransformerDataset, TransformerModel


def run(settings, transformer_path):
    now = datetime.now()
    seed = now.hour * 10000 + now.minute * 100 + now.second
    print("Using seed:", seed)
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    reader = Reader(settings, settings.input_path)
    tasks = reader.check_tasks(expected=None)

    label_encoder = MultiLabelEncoder.from_settings(settings, tasks=tasks)
    label_encoder.fit_reader(reader)
    if settings.verbose:
        print("::: Tasks :::")
        print()
        for task, le in label_encoder.tasks.items():
            print("- {:<15} target={:<6} level={:<6} vocab={:<6}"
                  .format(task, le.target, le.level, len(le)))
        print()

    tokenizer = AutoTokenizer.from_pretrained(transformer_path)
    transformer = AutoModel.from_pretrained(transformer_path)
    trainset = TransformerDataset(
        settings, reader, label_encoder, tokenizer, transformer)
    devset = None
    if settings.dev_path:
        devset = TransformerDataset(
            settings, Reader(settings, settings.dev_path), label_encoder,
            tokenizer, transformer)
    else:
        logging.warning("No devset: cannot monitor/optimize training")

    model = TransformerModel(
        label_encoder, settings.tasks, trainset.model.config.hidden_size,
        wemb_dim=settings.wemb_dim, cemb_dim=settings.cemb_dim,
        cemb_type=settings.cemb_type,
        custom_cemb_cell=settings.custom_cemb_cell,
        cemb_layers=settings.cemb_layers, cell=settings.cell,
        init_rnn=settings.init_rnn, merge_type=settings.merge_type,
        linear_layers=settings.linear_layers, dropout=settings.dropout,
        scorer=settings.scorer)
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
    trainer = Trainer(settings, model, trainset, reader.get_nsents())
    scores = None
    try:
        scores = trainer.train_epochs(settings.epochs, devset=devset)
    except KeyboardInterrupt:
        print("Stopping training")
    finally:
        model.eval()

    if devset is not None:
        scorers = model.evaluate(
            devset, trainset=trainset, use_beam=True, beam_width=10)
        for task, scorer in scorers.items():
            print(task)
            scorer.print_summary()
            print()

    fpath, infix = get_fname_infix(settings)
    if not settings.run_test:
        settings['transformer_path'] = os.path.join(
            os.getcwd(), transformer_path)
        fpath = model.save(fpath, infix=infix, settings=settings)
        print("Saved best model to: [{}]".format(fpath))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path')
    parser.add_argument('transformer_path')
    parser.add_argument('--opt_path', help='Path to optimization file (see opt.json)')
    parser.add_argument('--n_iter', type=int, default=20)
    # eval arguments
    parser.add_argument('--run_eval', action='store_true')
    parser.add_argument('--model_path', help='only used for evaluation')
    parser.add_argument('--test_path', help='only used for evaluation')
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--buffer_size', type=int, default=100000)
    parser.add_argument('--use_beam', action='store_true')
    parser.add_argument('--beam_width', type=int, default=12)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    settings = settings_from_file(args.config_path)

    if args.run_eval:
        model = TransformerModel.load(args.model_path)
        m_settings = model._settings
        m_settings.device = args.device
        m_settings.shuffle = False
        m_settings.batch_size = args.batch_size
        m_settings.buffer_size = args.buffer_size

        tokenizer = AutoTokenizer.from_pretrained(args.transformer_path)
        transformer = AutoModel.from_pretrained(args.transformer_path)
        trainset = TransformerDataset(
            m_settings, Reader(m_settings, m_settings.input_path), model.label_encoder,
            tokenizer, transformer)
        testset = TransformerDataset(
            m_settings, Reader(m_settings, args.test_path), model.label_encoder,
            tokenizer, transformer)

        for task in model.evaluate(
                testset, trainset, use_beam=args.use_beam, beam_width=args.beam_width
        ).values():
            task.print_summary()

    elif args.opt_path:
        opt = optimize.read_opt(args.opt_path)
        optimize.run_optimize(
            run, settings, opt, args.n_iter, transformer_path=args.transformer_path)
    else:
        run(settings, args.transformer_path)


# settings = settings_from_file('../pie/transformer-lemma.json')
# reader = Reader(settings, '../pie/datasets/capitula_classic_split/train0.train.tsv')
# label_encoder = MultiLabelEncoder.from_settings(settings).fit_reader(reader)
# trans_path = '../latin-data/latin-model/v4/checkpoint-110000'
# tokenizer = AutoTokenizer.from_pretrained(trans_path)
# transformer = AutoModel.from_pretrained(trans_path)
# trainset = pie.models.TransformerDataset(
#     settings, reader, label_encoder, tokenizer, transformer)
# model = pie.models.TransformerModel(
#     label_encoder, settings.tasks, trainset.model.config.hidden_size,
#     wemb_dim=settings.wemb_dim, cemb_dim=settings.cemb_dim,
#     cemb_type=settings.cemb_type,
#     custom_cemb_cell=settings.custom_cemb_cell,
#     cemb_layers=settings.cemb_layers, cell=settings.cell,
#     init_rnn=settings.init_rnn, merge_type=settings.merge_type,
#     linear_layers=settings.linear_layers, dropout=settings.dropout,
#     scorer=settings.scorer)
