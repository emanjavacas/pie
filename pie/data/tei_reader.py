
import re
import collections

from lxml import etree

from .base_reader import BaseReader, LineParseException


TEI = 'http://www.tei-c.org/ns/1.0'


def get_tei_tag(item):
    "Scrap namespace from tag"
    return re.sub("{{{}}}".format(TEI), "", item.tag)


def parse_pos(pos, simplify=False):
    "eventually simplify the pos tag removing subclass specs"
    if simplify:
        return re.sub(r"\([^\)]+\)", "", pos)
    return pos


class TEIReader(BaseReader):
    """
    TEI reader for BaB files
    """
    def __init__(self, settings, fpath):
        super(TEIReader, self).__init__(settings, fpath)

        self.max_sent_len = settings.max_sent_len

    def parselines(self):
        tree = etree.parse(self.fpath)
        inp, tasks_data = [], collections.defaultdict(list)

        for num, item in enumerate(tree.find('//tei:s', namespaces={'tei': TEI})):
            # avoid too long sentences
            if len(inp) >= self.max_sent_len:
                yield inp, dict(tasks_data)
                inp, tasks_data = [], collections.defaultdict(list)

            # check item tag
            tag = get_tei_tag(item)

            # ignore (physical line-breaks, deletions, editorials)
            if tag in ('lb', 'del', 'seg'):
                continue

            # punctuation
            elif tag == 'pc':
                inp.append(item.text.strip())
                tasks_data['pos'].append('PUNC')
                tasks_data['lemma'].append(item.text.strip())
                # # TODO: <pc> should segment nice sentences but it doesn't seem to
                # yield inp, dict(tasks_data)
                # inp, tasks_data = [], collections.defaultdict(list)

            # words
            elif tag == 'w':
                if item.text is None:
                    # <w lemma="" pos="" misAlignment="true:1/0" changed="yes"/>
                    continue
                inp.append(item.text.strip())
                pos = parse_pos(item.attrib['pos'], simplify=True)
                tasks_data['pos'].append(pos)
                lemma = item.attrib['lemma']
                tasks_data['lemma'].append(lemma)

            # unknown
            else:
                yield LineParseException(
                    "Encountered unexpected tag: {} at position #{}, file {}".format(
                        tag, num, self.fpath))
                inp, tasks_data = [], collections.defaultdict(list)

        # yield rest
        if len(inp) > 0:
            yield inp, tasks_data

    def get_tasks(self):
        return ('lemma', 'pos')
