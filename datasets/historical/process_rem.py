
import shutil
import os
from lxml import etree
import random
random.seed(100101)


root = 'rem-corralled-20161222'
target = 'rem'


def readlines(path):
    with open(path, 'rb') as f:
        tree = etree.fromstring(f.read()).getroottree()
        for token in tree.findall('token'):
            # there are 8 cases were tok_dipl is empty (resort to trans)
            form = token.find('tok_dipl').attrib['utf'] or token.attrib['trans']
            anno = token.find('tok_anno')
            pos = anno.find('pos').attrib['tag']
            if token.attrib['type'] == 'punc':
                # punctuation
                lemma = form
            else:
                lemma = anno.find('lemma').attrib['tag']
            # there is 1 case where lemma has whitespace: "nâh sup"
            if len(lemma.split()) > 1:
                print("Substituting complex lemma:", lemma)
                lemma = lemma.split()[0]

            yield form, lemma, pos


def make_subcorpus(subcorpus, **criteria):
    files = []
    for f in os.listdir(root):
        if not f.endswith('xml'):
            continue

        with open(os.path.join(root, f), 'rb') as inf:
            header = etree.fromstring(inf.read()).getroottree().find('header')

        is_in = True
        for crit, val in criteria.items():
            if header.find(crit).text != val:
                is_in = False
        if is_in:
            files.append(f)

    random.shuffle(files)
    five, ten = int(len(files) * 0.05), int(len(files) * 0.1)
    files = {
        'test': files[:ten],
        'dev': files[ten: ten+five],
        'train': files[ten+five:]
    }
    for split, files in files.items():
        subcorpuspath = os.path.join(target, subcorpus, 'splits', split)
        if not os.path.isdir(subcorpuspath):
            os.makedirs(subcorpuspath)

        with open(os.path.join(target, subcorpus, split + ".tsv"), 'w') as outf:
            outf.write('token\tlemma\tpos\n')
            for f in files:
                shutil.copy(
                    os.path.join(root, f),
                    os.path.join(target, subcorpus, 'splits', split, f))
                for form, lemma, pos in readlines(os.path.join(root, f)):
                    outf.write("{}\t{}\t{}\n".format(form, lemma, pos))


if __name__ == '__main__':
    make_subcorpus("poesie", topic="Poesie")
    make_subcorpus("recht", topic="Recht")
    make_subcorpus("religion", topic="Religion")

# count words per category
# counters = {k: Counter() for k in 'topic text-type genre language language-type language-region language-area time corpus'.split()}

# for f in files:
#     # nwords = len(list(readlines(os.path.join(root, f))))
#     with open(os.path.join(root, f), 'rb') as f:
#         header = etree.fromstring(f.read()).getroottree().find('header')
#         for k in counters:
#             counters[k][header.find(k).text] += 1
