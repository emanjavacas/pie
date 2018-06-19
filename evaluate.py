
import yaml

from pie import utils
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
    parser.add_argument('--model_info', action='store_true')
    args = parser.parse_args()

    model = BaseModel.load(args.model_path).to(args.device)
    if args.model_info:
        print(model)

    if hasattr(model, '_settings'):  # new models should all have _settings
        settings = model._settings
    elif args.settings_path:
        with utils.shutup():
            settings = settings_from_file(args.settings_path)
    else:
        with utils.shutup():
            settings = load_default_settings()

    # overwrite defaults
    settings.batch_size = args.batch_size
    settings.buffer_size = args.buffer_size
    settings.device = args.device

    reader = Reader(settings, args.test_path)
    dataset = Dataset(settings, reader, model.label_encoder)
    dataset = device_wrapper(list(dataset.batch_generator()), args.device)

    for task in model.evaluate(dataset).values():
        task.print_summary(full=args.full)
