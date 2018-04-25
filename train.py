import os
from pie.utils import Settings, settings_from_file
from pie.data import Dataset

s = settings_from_file(os.path.abspath('config.json'))
data = Dataset(s)

for buff in data.buffers(max_files=2):
    print()
    for sentence in buff:
        print(sentence)
