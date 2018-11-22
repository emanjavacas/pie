
from lxml import etree
import os
import random
random.seed(1001)

root = 'bab_versie_2.1alpha'
# make devsplit
train = os.listdir(os.path.join(root, 'train'))
random.shuffle(train)


TEI = 'http://www.tei-c.org/ns/1.0'
UNRESOLVED = 'UNRESOLVED'       # it's already use for complex cases


def readlines(path):
    with open(path) as f:
        tree = etree.fromstring(f.read()).getroottree()
        for w in tree.findall('//tei:w', namespaces={'tei': TEI}):
            if "misAlignment" in w.attrib:
                # continue
                continue
            token, lemma, pos = w.text, w.attrib['lemma'], w.attrib['pos']
            # resort to the token for numbers (otherwise you get full number in chars)
            if pos == 'TW' and token.isdecimal():
                lemma = token

            # some few tokens have weird whitespace: "' '"
            token = token.replace(' ', '_')

            # some few cases have no lemma, use token instead
            lemma = lemma or UNRESOLVED
            if ' ' in lemma:
                # complex cases (mostly proper names and lemmas with question marks)
                lemma = UNRESOLVED
            # some few cases have no pos, use UNK instead
            pos = pos or UNRESOLVED
            yield '{}\t{}\t{}\n'.format(token, lemma, pos)


five = int(len(train) * 0.05)
for split, files in {'dev': train[:five], 'train': train[five:]}.items():
    with open('{}/{}.tsv'.format(root, split), 'w') as f:
        f.write('token\tlemma\tpos\n')
        for inf in files:
            # print("***")
            # print(inf)
            for line in readlines(os.path.join(root, 'train', inf)):
                f.write(line)


# process test files
with open('{}/test.tsv'.format(root), 'w') as f:
    f.write('token\tlemma\tpos\n')
    for inf in os.listdir(os.path.join(root, 'test')):
        for line in readlines(os.path.join(root, 'test', inf)):
            f.write(line)
