# Can be run with python -m pie.scripts.train
import time
import logging


from pie.settings import settings_from_file
from pie.trainer import Trainer, get_fname_infix, set_seed, get_targets
from pie.data import Dataset, Reader
from pie import optimize


def run(settings):
    seed = set_seed(verbose=settings.verbose)
    if settings.verbose:
        logging.basicConfig(level=logging.INFO)

    trainer, reader, devset = Trainer.setup(settings)
    model = trainer.model
    trainset = trainer.dataset
    label_encoder = model.label_encoder

    # training
    print("Starting training")
    running_time = time.time()
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
