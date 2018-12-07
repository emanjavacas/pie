
import os
from lxml import etree
import random
random.seed(100101)

root = 'ReN_2018-07-23'
target = 'ren'
if not os.path.isdir(target):
    os.makedirs(target)


def readlines(path):
    with open(path, 'rb') as f:
        tree = etree.fromstring(f.read()).getroottree()
        for token in tree.findall('token'):
            # there are 8 cases were tok_dipl is empty (resort to trans)
            form = token.find('dipl').attrib['utf'] or token.attrib['trans']
            anno = token.find('anno')
            pos = anno.find('pos').attrib['tag']
            lemma = anno.find('lemma').attrib['tag']
            morph = anno.find('morph').attrib['tag']
            # there are 3 cases where lemma has whitespace: "wager man"
            if len(lemma.split()) > 1:
                print("Substituting complex lemma:", lemma)
                lemma = lemma.split()[0]
            # there are 5 cases with empty lemma
            if not lemma:
                lemma = "<none>"

            yield form, lemma, pos, morph


def writelines(f, lines):
    for token, lemma, pos, morph in lines:
        f.write("{}\t{}\t{}\t{}\n".format(token, lemma, pos, morph))


files = os.listdir(root)
random.shuffle(files)
formatter = os.path.join(target, '{}.tsv').format
with open(formatter('train'), 'w+') as train, \
     open(formatter('test'), 'w+') as test, \
     open(formatter('dev'), 'w+') as dev:
    train.write('token\tlemma\tpos\tmorph\n')
    test.write('token\tlemma\tpos\tmorph\n')
    dev.write('token\tlemma\tpos\tmorph\n')
    for f in files:
        lines = list(readlines(os.path.join(root, f)))
        if len(lines) < 5000:
            writelines(train, lines)
        else:
            five = int(len(lines) * 0.05)
            ten = int(len(lines) * 0.1)
            writelines(dev, lines[:ten])
            writelines(test, lines[ten: five + (2*ten)])
            writelines(train, lines[five + (2*ten):])
