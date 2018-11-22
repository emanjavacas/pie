
import os
from lxml import etree
import random
random.seed(1001)

TEI = 'http://www.tei-c.org/ns/1.0'

root = 'goo300k'
if not os.path.isdir(root):
    os.makedirs(root)
path = './Reference corpus of historical Slovene goo300k 1.2/goo300k-vert/goo300k.vert'
with open(path) as f:
    tree = etree.fromstring(f.read().replace('<g/>', '')).getroottree()

sents = tree.findall('//tei:s', namespaces={'tei': TEI})


def process_sent(sent):
    for line in sent.text.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        token, _, lemma, pos, *_ = line.split('\t')
        yield token, lemma, pos


random.shuffle(sents)
five = int(len(sents) * 0.05)

for split, sents in {'dev':   sents[:five],
                     'test':  sents[five:(five*2)+five],
                     'train': sents[(five*2)+five:]}.items():
    with open(os.path.join(root, '{}.tsv'.format(split)), 'w') as f:
        f.write('{}\t{}\t{}\n'.format("token", "lemma", "pos"))
        for sent in sents:
            line = list(process_sent(sent))
            if len(line) == 0:
                print("Empty line")
                continue
            for token, lemma, pos in line:
                f.write('{}\t{}\t{}\n'.format(token, lemma, pos))
            f.write('\n')
