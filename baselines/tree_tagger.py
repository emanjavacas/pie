
import treetagger

from pie import compute_scores
from pie.settings import load_default_settings
from pie.data import Reader

settings = load_default_settings()
settings.input_path = 'datasets/capitula_classic_split/train0.dev.tsv'
reader = Reader(settings, settings.input_path)

trues = []
preds = []

tagger = treetagger.TreeTagger(language="latin", path_to_treetagger="/home/manjavacas/code/vendor/treetagger/")

for idx, line in enumerate(reader.readsents(only_tokens=True)):
    if idx and idx % 100:
        print(".")

    for t, p in zip(line, tagger.tag(line)):
        p = p[-1]
        # sometimes treetagger outputs multiple lemmata
        if "|" in p:
            for pp in p.split("|"):
                if pp == t:
                    p = t
        trues.append(t)
        preds.append(p)

print(compute_scores(trues, preds))
