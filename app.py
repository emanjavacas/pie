"""

This module can be run by install the app requirements (`pip install -r requirements-app.txt`)

The goal here is to be able to have, later, a distributed series of APIs that could be using different version of Pie
and run through containers, unified by a public API such as https://github.com/hipster-philology/deucalion which would
then talk to the different micro-services.

How to run for development :
    PIE_MODEL=/home/thibault/dev/pie/model-lemma-2018_10_23-14_05_19.tar FLASK_ENV=development flask run

    where PIE_MODEL is the path to your model

How to run in production :
    gunicorn --workers 2 app:app --env PIE_MODEL=/home/thibault/dev/pie/model-lemma-2018_10_23-14_05_19.tar

    Probably add to this a --bind

Example URL:
    http://localhost:5000/?data=Ci+gist+saint+Martins+el+sains+de+tours.%20Il%20fut%20bon%20rois.

Example curl :
    curl --data "data=Ci gist saint Martins el sains de tours. Il fut bon rois." http://localhost:5000

Example output :
    token	lemma	morph	pos
    ci	ci	DEGRE=-	ADVgen
    gist	jesir	MODE=ind|TEMPS=pst|PERS.=3|NOMB.=s	VERcjg
    saint	saint	NOMB.=s|GENRE=f|CAS=r	ADJqua
    martins	martin	NOMB.=s|GENRE=m|CAS=r	NOMcom
    el	en1+le	NOMB.=s|GENRE=m|CAS=r	PRE.DETdef
    sains	sain	NOMB.=p|GENRE=m|CAS=r	NOMcom
    de	de	MORPH=empty	PRE
    tours	tor2	NOMB.=p|GENRE=f|CAS=r	NOMcom
    .	.	_	PONfrt
    il	il	PERS.=3|NOMB.=s|GENRE=m|CAS=n	PROper
    fut	estre1	MODE=ind|TEMPS=psp|PERS.=3|NOMB.=s	VERcjg
    bon	bon	NOMB.=s|GENRE=m|CAS=n|DEGRE=p	ADJqua
    rois	roi2	NOMB.=s|GENRE=m|CAS=n	NOMcom
    .	.	_	PONfrt

"""
from flask import Flask, request, Response, stream_with_context
from pie.tagger import Tagger, simple_tokenizer
from pie.utils import chunks, model_spec
from os import getenv

model_file = getenv("PIE_MODEL")
BATCH = int(getenv("PIE_BATCH", 3))
DEVICE = getenv("PIE_DEVICE", "cpu")

app = Flask(__name__)
tagger = Tagger(device=DEVICE, batch_size=BATCH)


for model, tasks in model_spec(model_file):
    tagger.add_model(model, *tasks)
    tasks = tasks or tagger.models[-1][0].label_encoder.tasks


def iter_data(data, lower=False):
    for sentence in simple_tokenizer(data, lower=lower):
        yield sentence, len(sentence)


@app.route("/", methods=["POST", "GET", "OPTIONS"])
def index():
    def lemmatization_stream():
        lower = request.args.get("lower", False)
        if lower:
            lower = True

        if request.method == "GET":
            data = request.args.get("data")
        else:
            data = request.form.get("data")

        if not data:
            yield ""

        header = False
        for chunk in chunks(iter_data(data, lower=lower), size=BATCH):
            sents, lengths = zip(*chunk)

            tagged, tasks = tagger.tag(sents=sents, lengths=lengths)
            sep = "\t"
            for sent in tagged:
                if not header:
                    yield sep.join(['token'] + tasks) + '\r\n'
                    header = True
                for token, tags in sent:
                    yield sep.join([token] + list(tags)) + '\r\n'

    return Response(
           stream_with_context(lemmatization_stream()),
           200,
           headers={
            'Content-Type': 'text/plain',
            'Access-Control-Allow-Origin': '*'
           }
    )
