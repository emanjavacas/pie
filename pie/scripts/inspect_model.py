
import pie
import yaml

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('modelpath')
    args = parser.parse_args()

    m = pie.SimpleModel.load(args.modelpath)
    print("::: Settings :::")
    print(yaml.dump(dict(m._settings)))
    print()
    print("::: Model :::")
    print(m)

