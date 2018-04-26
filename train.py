
import os

from pie.settings import settings_from_file
from pie.data import Dataset

settings = settings_from_file(os.path.abspath('config.json'))
data = Dataset(settings)

for batch in data.batches():
    print()
    for sent in batch:
        print(sent)
