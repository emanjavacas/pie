
import collections
from difflib import SequenceMatcher


def segment(s, infix):
    idx = s.find(infix)
    return s[:idx], s[idx+len(infix):]


def get_segments(a, b, match):
    infix = a[match.a: match.a + match.size]
    return segment(a, infix), segment(b, infix)


def lcs(a, b):
    return SequenceMatcher(None, a, b).find_longest_match(0, len(a), 0, len(b))


def breadth_first(root):
    def breadth_first_(level, *nodes):
        children = []
        for node in nodes:
            yield level, node
            children.extend(node.children())
        if children:
            yield from breadth_first_(level+1, *children)

    return list(breadth_first_(1, root))


def depth_first(root):
    def depth_first_(level, node):
        yield level, node
        for child in node.children():
            yield from depth_first_(level+1, child)

    return list(depth_first_(1, root))


class Node:
    def __init__(self, _prefix, _infix, _suffix, target):
        self._prefix = _prefix  # prefix of the string at this level
        self._infix = _infix    # current string
        self._suffix = _suffix  # suffix of the string at this level
        self.target = target
        # children
        self.prefix = None
        self.suffix = None

    def __len__(self):
        return len(self._infix)

    def __repr__(self):
        tree = ""
        for level, child in depth_first(self):
            tree += ' ' * (level - 1) * 2
            if isinstance(child, Leaf):
                tree += '- {}\n'.format(child)
            else:
                tree += '{}({},{})\n'.format(child._infix, *child.get_span())
        return tree

    def get_source(self):
        return self._prefix + self._infix + self._suffix

    def get_span(self):
        return len(self._prefix), len(self._suffix)

    def get_depth(self):
        return max(1 + self.prefix.get_depth() if self.prefix else 0,
                   1 + self.suffix.get_depth() if self.suffix else 0)

    def children(self):
        return [self.prefix, self.suffix]

    def to_class(self):
        return get_class(self)

    def to_tuple(self):
        if isinstance(self, Leaf):
            # full replacement
            return (len(self.a), self.b, 0, "")
        else:
            plen, slen = self.get_span()
            start = self.target.find(self._infix)
            pstring, sstring = self.target[:start], self.target[start+len(self._infix):]
            return (plen, pstring, slen, sstring)


class Leaf(Node):
    def __init__(self, a, b):
        self.a = a
        self.b = b
        # cache rule and string representation
        self.string = self.rule = None

        if a and b:
            self.rule = Rule('replace', None, b)
            self.string = '<replace({})>'.format(b)
        elif a:
            self.rule = Rule('delete', a, None)
            self.string = '<delete({})>'.format(a)
        elif b:
            self.rule = Rule('insert', None, b)
            self.string = '<insert({})>'.format(b)
        else:
            self.rule = Rule('keep', None, None)
            self.string = '<keep>'

    def __repr__(self):
        return self.string

    def __len__(self):
        return 0

    def get_depth(self):
        return 1

    def children(self):
        return []


Rule = collections.namedtuple('Rule', ['action', 'a', 'b'])


def apply_rule(rule, inp, prefix=False):
    if rule.action == 'replace':
        return rule.b
    elif rule.action == 'delete':
        return inp[len(rule.a):] if prefix else inp[:-len(rule.a)]
    elif rule.action == 'insert':
        return (rule.b + inp) if prefix else (inp + rule.b)
    else:
        return inp


def get_class(node):
    if isinstance(node, Leaf):
        return node.rule
    else:
        return node.get_span(), (get_class(node.prefix), get_class(node.suffix))


def make_edit_tree(a, b):
    match = lcs(a, b)

    # check if leaf
    if match.size == 0:
        return Leaf(a, b)

    # parent
    else:
        (pre_a, suf_a), (pre_b, suf_b) = get_segments(a, b, match)
        node = Node(pre_a, a[match.a:match.a+match.size], suf_a, b)
        node.prefix = make_edit_tree(pre_a, pre_b)
        node.suffix = make_edit_tree(suf_a, suf_b)
        return node


def apply_edit_tree(tclass, inp, prefix=False):
    if isinstance(tclass, Rule):
        # apply leaf
        return apply_rule(tclass, inp, prefix)
    else:
        # split and recur on non-terminals
        (plen, slen), (p, s) = tclass
        infix = inp[plen:len(inp)-slen]
        prefix = apply_edit_tree(p, inp[:plen], prefix=True)
        suffix = apply_edit_tree(s, inp[len(inp)-slen:])
        return prefix + infix + suffix


def apply_tuple(tup, inp):
    plen, pstring, slen, sstring = tup
    return pstring + inp[plen:-slen] + sstring


def transform(lem, tok):
    return make_edit_tree(tok, lem).to_class()


def inverse_transform(pred, tok):
    return apply_edit_tree(pred, tok)
