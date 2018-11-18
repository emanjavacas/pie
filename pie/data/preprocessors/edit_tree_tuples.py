
from .edit_trees import make_edit_tree


def transform(lem, tok):
    return make_edit_tree(tok, lem).to_tuple()


def inverse_transform(pred, tok):
    plen, pstring, slen, sstring = pred
    return pstring + tok[plen:-slen or None] + sstring
