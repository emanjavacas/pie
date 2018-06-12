
import os
import glob
import itertools
import re


def window(it):
    """
    >>> list(window(range(5)))
    [(None, 0, 1), (0, 1, 2), (1, 2, 3), (2, 3, 4), (3, 4, None)]
    """
    it = itertools.chain([None], it, [None])  # pad for completeness
    result = tuple(itertools.islice(it, 3))

    if len(result) == 3:
        yield result

    for elem in it:
        result = result[1:] + (elem,)
        yield result


def chunks(it, size):
    """
    Chunk a generator into a given size (last chunk might be smaller)
    """
    buf = []
    for s in it:
        buf.append(s)
        if len(buf) == size:
            yield buf
            buf = []
    if len(buf) > 0:
        yield buf


def flatten(it):
    """
    >>> list(flatten([['abc', 'cde'], ['yte']]))
    ['a', 'b', 'c', 'c', 'd', 'e', 'y', 't', 'e']
    """
    if isinstance(it, str):
        for i in it:
            yield i
    else:
        for subit in it:
            yield from flatten(subit)


def ensure_ext(path, ext, infix=None):
    """
    Compute target path with eventual infix and extension

    >>> ensure_ext("model.pt", "pt", infix="0.87")
    'model-0.87.pt'
    >>> ensure_ext("model.test", "pt", infix="0.87")
    'model-0.87.test.pt'
    """
    path, oldext = os.path.splitext(path)
    if oldext.startswith("."):
        oldext = oldext[1:]
    if infix is not None:
        path = "-".join([path, infix])
    if oldext != ext:
        path = '.'.join([path, oldext])

    return '.'.join([path, ext])


def get_filenames(input_path):
    """
    Get filenames from path expression
    """
    if os.path.isdir(input_path):
        filenames = [os.path.join(input_path, f) for f in os.listdir(input_path)
                     if not f.startswith('.')]
    elif os.path.isfile(input_path):
        filenames = [input_path]
    else:
        filenames = glob.glob(input_path)

    if len(filenames) == 0:
        raise RuntimeError("Couldn't find files [{}]".format(input_path))

    return filenames
