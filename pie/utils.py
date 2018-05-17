
import itertools


def window(it):
    """
    >>> list(window(range(10)))
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


