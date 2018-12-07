
import os
from lxml.etree import XMLParser, parse
import random
random.seed(1001)

ALDT = 'http://ufal.mff.cuni.cz/pdt/pml/'

parser = XMLParser(huge_tree=True)
target = 'LLCT1'
root = './LLCT1.xml'
roottree = parse(root, parser=parser)

OMIT = set('id relation seg status form lemma pos'.split())


def parsetree(tree):
    sent = []
    for node in sorted(tree.iterdescendants(), key=lambda node: int(node.attrib['id'])):
        token, lemma = node.attrib['form'], node.attrib['lemma']
        if 'pos' not in node.attrib:
            pos = 'punc'
        else:
            pos = node.attrib['pos']
        if ' ' in token:        # some tokens are like ".... ...."
            token = '...'
        assert token
        assert lemma
        assert pos
        assert " " not in token
        assert " " not in lemma
        assert " " not in pos
        morph = '|'.join('{}={}'.format(k, v)
                         for k, v in sorted(node.attrib.items()) if k not in OMIT)
        sent.append((token, lemma, pos, morph))
    return sent


def parsetrees(roottree):
    for tree in roottree.xpath(
            '//aldt:LM[not(ancestor::aldt:LM)]', namespaces={'aldt': ALDT}):
        yield parsetree(tree)


if __name__ == '__main__':
    # trees = roottree.xpath('//aldt:LM[not(ancestor::aldt:LM)]', namespaces={'aldt': ALDT})
    # import collections
    # counts = {k: collections.Counter()
    #           for k in 'document_id subdoc date place scribe type'.split()}
    # for tree in trees:
    #     for k in counts:
    #         counts[k][tree.attrib[k]] += 1

    trees = list(parsetrees(roottree))
    random.shuffle(trees)
    five = int(len(trees) * 0.05)
    if not os.path.isdir(target):
        os.makedirs(target)
    for split, sents in {'dev': trees[:five],
                         'test': trees[five:3*five],
                         'train': trees[3*five:]}.items():
        with open(os.path.join(target, '{}.tsv'.format(split)), 'w+') as f:
            f.write('token\tlemma\tpos\tmorph\n')
            for sent in sents:
                for token, lemma, pos, morph in sent:
                    f.write('{}\t{}\t{}\t{}\n'.format(token, lemma, pos, morph))
                f.write('\n')
