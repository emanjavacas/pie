"""
    For more information about how to run the webapp, go check app.py in
    the root directory.
    This module is available so that you can enhance or customize your app
    by changing its tokenizer for example

"""
from os import getenv
from typing import Iterator, List, Tuple, Callable, Iterable
from flask import Flask, request, Response, stream_with_context, current_app

from pie.tagger import Tagger, simple_tokenizer
from pie.utils import chunks, model_spec

MODEL_FILE = getenv("PIE_MODEL")
BATCH = int(getenv("PIE_BATCH", 3))
DEVICE = getenv("PIE_DEVICE", "cpu")


Tokenizer = Callable[[str, bool], Iterable[List[str]]]


class DataIterator:
    def __init__(self, tokenizer: Tokenizer = None):
        """ Iterator used to parse the text and returns bits to tag

        :param tokenizer: Tokenizer
        """
        self.tokenizer = tokenizer or simple_tokenizer

    def __call__(self, data: str, lower: bool = False) -> Iterable[Tuple[List[str], int]]:
        """ Default iter data takes a text, an option to make lower
        and yield lists of words along with the length of the list

        :param data: A plain text
        :param lower: Whether or not to lower the text
        :yields: (Sentence as a list of word, Size of the sentence)
        """
        for sentence in self.tokenizer(data, lower=lower):
            yield sentence, len(sentence)


class Formatter:
    def __init__(self, tasks: List[str]):
        self.tasks = tasks

    def format_headers(self)-> List[str]:
        """ Format the headers """
        return ["token"] + self.tasks

    def format_line(self, token: str, tags: Iterable[str]) -> List[str]:
        """ Format the tags"""
        return [token] + list(tags)


def bind(app: Flask = None, device: str = None, batch_size: int = None,
         model_file: str = None, formatter_class: Formatter = None,
         tokenizer: Tokenizer = None, allow_origin: str = None,
         route_path: str= "/", headers=None) -> Flask:
    """ Binds default value

    :param app: Flask app to bind with the tagger
    :param device: Device to use for PyTorch (Default : cpu)
    :param batch_size: Size of the batch to treat
    :param model_file: Model to use to tag
    :param formatter_class: Formatter of response
    :param tokenizer: Tokenizer to split text into segments (eg. sentence) and into words
    :param allow_origin: Value for the http header field Access-Control-Allow-Origin
    :param route_path: Route for the API (default : `/`)
    :param headers: Additional headers
    :returns: Application
    """
    # Generates or use default values for non completed parameters
    if not app:
        app = Flask(__name__)

    if not batch_size:
        batch_size = BATCH

    device = device or DEVICE
    model_file = model_file or MODEL_FILE
    allow_origin = allow_origin or '*'

    if tokenizer:
        data_iterator = DataIterator(tokenizer)
    else:
        data_iterator = DataIterator()

    tagger = Tagger(device=device, batch_size=batch_size)

    for model, tasks in model_spec(model_file):
        tagger.add_model(model, *tasks)

    formatter_class = formatter_class or Formatter

    _headers = {
        'Content-Type': 'text/plain; charset=utf-8',
        'Access-Control-Allow-Origin': allow_origin
       }
    if headers:
        _headers.update(headers)

    @app.route(route_path, methods=["POST", "GET", "OPTIONS"])
    def lemmatize():
        def lemmatization_stream() -> Iterator[str]:
            lower = request.args.get("lower", False)
            if lower:
                lower = True

            if request.method == "GET":
                data = request.args.get("data")
            else:
                data = request.form.get("data")

            if not data:
                yield ""
            else:
                header = False
                for chunk in chunks(data_iterator(data, lower=lower), size=BATCH):
                    sents, lengths = zip(*chunk)

                    tagged, tasks = tagger.tag(sents=sents, lengths=lengths)
                    formatter = formatter_class(tasks)
                    sep = "\t"
                    for sent in tagged:
                        if not header:
                            yield sep.join(formatter.format_headers()) + '\r\n'
                            header = True
                        for token, tags in sent:
                            yield sep.join(formatter.format_line(token, tags)) + '\r\n'

        return Response(
               stream_with_context(lemmatization_stream()),
               200,
               headers=_headers
        )

    return app
