
import os

from pie.settings import settings_from_file
from pie.data import Dataset

settings = settings_from_file(os.path.abspath('config.json'))
data = Dataset(settings)
print(next(data.batch_generator()))

# for batch in data.batch_generator():
#     print()
#     for sent in batch:
#         print(sent)
