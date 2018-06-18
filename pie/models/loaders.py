
import numpy as np


def get_pretrained_embeddings(reader, label_encoder, **kwargs):
    from pie import utils
    with utils.shutup():        # avoid pattern warning
        from gensim.models import Word2Vec

    word2vec = Word2Vec(reader.get_token_iterator(), **kwargs)
    weight = np.zeros((len(label_encoder.word), word2vec.wv.vector_size))

    found = 0
    for w, idx in label_encoder.word.table.items():
        try:
            weight[idx] = word2vec.wv[w]
            found += 1
        except KeyError:  # reserved symbols are not in training sentences
            pass

    print("A total of {}/{} word embeddings were pretrained"
          .format(found, len(label_encoder.word)))

    return weight


if __name__ == '__main__':
    from pie.data import Reader, MultiLabelEncoder
    from pie.settings import settings_from_file

    settings = settings_from_file("config.json")
    reader = Reader(settings, settings.input_path)
    le = MultiLabelEncoder.from_settings(settings)
    le.fit_reader(reader)
    get_pretrained_embeddings(reader, le, min_count=1)
