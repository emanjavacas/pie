import os
from copy import deepcopy
import glob
import random
random.seed(12345)

empty = {'tokens': [], 'pos': [], 'lemma': []}

class Dataset(object):
    def __init__(self, settings):
        super(Dataset, self).__init__()

        self.indir = os.path.abspath(settings.input_dir)
        self.filenames = glob.glob(self.indir + f'/*.{settings.extension}')
        self.buffer_size = settings.buffer_size # nb of sentences per buffer
        self.sentence_length = settings.sentence_length

    def buffers(self, max_files=None):
        random.shuffle(self.filenames)
        buff = []

        for filename in self.filenames[:max_files]:
            sent = deepcopy(empty)
            for line in open(filename, 'r'):
                try:
                    tok, lem, pos = line.strip().split()
                except ValueError:
                    continue

                sent['tokens'].append(tok)
                sent['pos'].append(pos)
                sent['lemma'].append(lem)

                if len(sent['tokens']) >= self.sentence_length:
                    buff.append({k : tuple(v) for k, v in sent.items()})
                    sent = deepcopy(empty)

                if len(buff) == self.buffer_size:
                    yield tuple(buff)
                    buff = []
            
            if buff:
                yield buff





