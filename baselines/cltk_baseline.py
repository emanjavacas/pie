
from cltk.stem import lemma
from cltk.corpus.utils.importer import CorpusImporter

corpus_importer = CorpusImporter('latin')
corpus_importer.import_corpus('latin_models_cltk')
lemmatizer = lemma.LemmaReplacer('latin')
lemmatizer.lemmatize("arma virumque cano".split())

from pie import compute_scores
from pie.settings import load_default_settings
from pie.data import Reader

settings = load_default_settings()
settings.input_path = 'datasets/capitula_classic_split/train0.dev.tsv'
reader = Reader(settings, settings.input_path)

trues = []
preds = []
for line in reader.readsents(only_tokens=True):
    for t, p in zip(line, lemmatizer.lemmatize(line)):
        trues.append(t)
        preds.append(p)

print(compute_scores(trues, preds))
