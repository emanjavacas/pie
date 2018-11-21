
import re
import os
import shutil
import uuid
import gzip
import logging
import sys
import glob
import itertools
from contextlib import contextmanager
from subprocess import check_output, CalledProcessError


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
    >>> ensure_ext("model.test", "test", infix="pie")
    'model-pie.test'
    """
    path, oldext = os.path.splitext(path)

    # normalize extension
    if ext.startswith("."):
        ext = ext[1:]
    if oldext.startswith("."):
        oldext = oldext[1:]

    # infix
    if infix is not None:
        path = "-".join([path, infix])

    # add old extension if not the same as the new one
    if oldext and oldext != ext:
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
    else:                       # unix string
        filenames = glob.glob(input_path)

    return filenames


@contextmanager
def shutup():
    with open(os.devnull, "w") as void:
        old_out = sys.stdout
        old_err = sys.stderr
        sys.stdout = void
        sys.stderr = void
        try:
            yield
        finally:
            sys.stdout = old_out
            sys.stderr = old_err


@contextmanager
def tmpfile(parent='/tmp/'):
    fid = str(uuid.uuid1())
    tmppath = os.path.join(parent, fid)
    yield tmppath
    if os.path.isdir(tmppath):
        shutil.rmtree(tmppath)
    else:
        os.remove(tmppath)


def add_gzip_to_tar(string, subpath, tar):
    with tmpfile() as tmppath:
        with gzip.GzipFile(tmppath, 'w') as f:
            f.write(string.encode())
        tar.add(tmppath, arcname=subpath)


def get_gzip_from_tar(tar, fpath):
    return gzip.open(tar.extractfile(fpath)).read().decode().strip()


class GitInfo():
    """
    Utility class to retrieve git-based info from a repository
    """
    def __init__(self, fname):
        if os.path.isfile(fname):
            self.dirname = os.path.dirname(fname)
        elif os.path.isdir(fname):
            self.dirname = fname
        else:
            # not a file
            self.dirname = None

        if not os.path.isfile(fname) and not os.path.isdir(fname):
            logging.warn("[GitInfo]: Input file doesn't exit")

        else:
            try:
                with shutup():
                    check_output(['git', '--version'], cwd=self.dirname)
            except FileNotFoundError:
                self.dirname = None
                logging.warn("[GitInfo]: Git doesn't seem to be installed")
            except CalledProcessError as e:
                self.dirname = None
                code, _ = e.args
                if code == 128:
                    logging.warn("[GitInfo]: Script not git-tracked")
                else:
                    logging.warn("[GitInfo]: Unrecognized git error")

    def run(self, cmd):
        if self.dirname is None:
            return

        return check_output(cmd, cwd=self.dirname).strip().decode('utf-8')

    def get_commit(self):
        """
        Returns current commit on file or None if an error is thrown by git
        (OSError) or if file is not under git VCS (CalledProcessError)
        """
        return self.run(["git", "describe", "--always"])

    def get_branch(self):
        """
        Returns current active branch on file or None if an error is thrown
        by git (OSError) or if file is not under git VCS (CalledProcessError)
        """
        return self.run(["git", "rev-parse", "--abbrev-ref", "HEAD"])

    def get_tag(self):
        """
        Returns current active tag
        """
        return self.run(["git", "describe", "--tags", "--abbrev=0"])


def model_spec(inp):
    """
    >>> example = 'model-pos-2018:03:05.tar'
    >>> model_spec(example)
    [('model-pos-2018:03:05.tar', [])]

    >>> example = '<model-pos-2018:03:05.tar,pos><model-pos-2018:03:05.tar,lemma>'
    >>> model_spec(example)
    [('model-pos-2018:03:05.tar', ['pos']), ('model-pos-2018:03:05.tar', ['lemma'])]
    """
    if not inp.startswith('<'):
        return [(inp, [])]

    output = []
    for string in re.findall(r'<([^>]+)>', inp):
        model_path, *tasks = string.split(',')
        output.append((model_path, tasks))

    return output
