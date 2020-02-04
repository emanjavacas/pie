
import pie
import yaml
import json

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('modelpath')
    parser.add_argument('--export_settings')
    args = parser.parse_args()

    m = pie.SimpleModel.load(args.modelpath)
    print("::: Settings :::")
    print(yaml.dump(dict(m._settings)))
    print()
    print("::: Model :::")
    print(m)

    if args.export_settings:
        with open(pie.utils.ensure_ext(args.export_settings, 'json'), 'w') as f:
            f.write(json.dumps(dict(m._settings), indent=2))
    
