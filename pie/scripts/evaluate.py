# Can be run with python -m pie.scripts.evaluate
import os.path
from pie import utils


from pie.models import BaseModel
from pie.data import Dataset, Reader
from pie.settings import load_default_settings, settings_from_file


def run(model_path, test_path, train_path,
        settings, batch_size, buffer_size, device, model_info, full, confusion,
        report, markdown):

    model = BaseModel.load(model_path).to(device)
    if model_info:
        print(model)

    if hasattr(model, '_settings'):  # new models should all have _settings
        settings = model._settings
    elif settings:
        with utils.shutup():
            settings = settings_from_file(settings)
    else:
        with utils.shutup():
            settings = load_default_settings()

    # overwrite defaults
    settings.batch_size = batch_size
    settings.buffer_size = buffer_size
    settings.device = device

    # label encoder
    DatasetClass = Dataset
    if settings.wemb_type == "transformer":
        from pie.data.transformer_dataset import TransformerDataset
        DatasetClass = TransformerDataset

    trainset = None
    if train_path:
        trainset = DatasetClass(
            settings, Reader(settings, train_path), model.label_encoder)
    elif hasattr(settings, "input_path") and settings.input_path and os.path.exists(settings.input_path):
        print("--- Using train set from settings")
        trainset = DatasetClass(
            settings, Reader(settings, settings.input_path), model.label_encoder)
    else:
        print("--- Not using trainset to evaluate known/unknown tokens")

    if not len(test_path) and hasattr(settings, "test_path"):
        print("--- Using test set from settings")
        test_path = (settings.test_path, )

    testset = DatasetClass(settings, Reader(settings, *test_path), model.label_encoder)

    for task in model.evaluate(testset, trainset).values():
        task.print_summary(full=full, confusion_matrix=confusion, report=report, markdown=markdown)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path')
    parser.add_argument('test_path', help="unix string", nargs='+')
    parser.add_argument('train_path',
                        help="needed to compute ambiguous and unknown tokens")
    parser.add_argument('--settings', help="settings file used for training")
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--buffer_size', type=int, default=100000)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--model_info', action='store_true')
    parser.add_argument('--full', action='store_true')
    parser.add_argument('--confusion', default=False, action="store_true")
    parser.add_argument('--report', default=False, action="store_true", help="Get full report on each class")
    parser.add_argument('--markdown', default=False, action="store_true", help="Use Markdown")
    args = parser.parse_args()
    run(model_path=args.model_path, test_path=args.test_path,
        train_path=args.train_path, settings=args.settings,
        batch_size=args.batch_size, buffer_size=args.buffer_size,
        device=args.device, model_info=args.model_info,
        full=args.full, confusion=args.confusion, report=args.report,
        markdown=args.markdown)
