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
# Can be run with python -m pie.scripts.tagger_pipe
from pie.webapp import bind


def run(device: str = None, batch_size: int = None, model_file: str = None):
    app = bind(device=device, batch_size=batch_size, model_file=model_file)
    return app


if __name__ == '__main__':
    run()
