
import yaml

from pie.models import BaseModel
from pie.data import Dataset, Reader, device_wrapper
from pie.settings import load_default_settings, settings_from_file


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path')
    parser.add_argument('test_path', help="unix string")
    parser.add_argument('--settings_path', help="settings file used for training")
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--buffer_size', type=int, default=100000)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    model = BaseModel.load(args.model_path)
    print(model)
    model.eval()
    model.to(args.device)

    if hasattr(model, '_settings'):
        settings = model._settings
    elif args.settings_path:
        settings = settings_from_file(args.settings_path)
    else:
        settings = load_default_settings()

    # overwrite defaults
    settings.batch_size = args.batch_size
    settings.buffer_size = args.buffer_size
    settings.device = args.device

    reader = Reader(settings, args.test_path)

    print("Loading dataset")
    dataset = Dataset(settings, reader, model.label_encoder)
    dataset = device_wrapper(list(dataset.batch_generator()), args.device)

    print("Evaluating on test set")
    scores = model.evaluate(dataset)
    print()
    print("::: Test scores :::")
    print()
    print(yaml.dump(scores, default_flow_style=False))
    print()
