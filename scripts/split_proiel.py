
import os
import glob


def get_sents(fpath):
    with open(fpath) as f:
        sent = []
        for line in f:
            if len(line.strip().split()) == 0:
                if len(sent) > 0:
                    yield sent
                    sent = []
            else:
                sent.append(line)


if __name__ == '__main__':
    inp = './datasets/proiel-treebank/src/'
    train = './datasets/proiel-treebank/train/'
    test = './datasets/proiel-treebank/test/'

    for f in glob.glob(inp + '*.conll'):
        sents = list(get_sents(f))
        split = int(len(sents) * 0.1)

        basename = os.path.basename(f)
        with open(os.path.join(train, basename), 'w+') as f:
            for sent in sents[:-split]:
                for line in sent:
                    f.write(line)
                f.write('\n')
        with open(os.path.join(test, basename), 'w+') as f:
            for sent in sents[-split:]:
                for line in sent:
                    f.write(line)
                f.write('\n')
