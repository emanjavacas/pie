
import os


def readlines(path):
    with open(path) as f:
        header = next(f).strip().split()
        sent = []
        for line in f:
            line = dict(zip(header, line.strip().split()))
            sent.append(line['token'])
            if line['pos'] == '$.':
                yield sent
                sent = []
        if sent:
            yield sent


def writelines(path):
    outpath = os.path.splitext(path)[0] + '.lines.txt'
    with open(outpath, 'w') as out:
        for sent in readlines(path):
            out.write(' '.join(sent) + '\n')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train')
    parser.add_argument('--test')
    args = parser.parse_args()

    writelines(args.train)
    writelines(args.test)
