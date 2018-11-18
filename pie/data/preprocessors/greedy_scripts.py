
import difflib
import collections


Keep = collections.namedtuple('Keep', [])
Replace = collections.namedtuple('Replace', ['source', 'target'])
Insert = collections.namedtuple('Insert', ['string', 'pos'])
Delete = collections.namedtuple('Delete', ['string'])


def get_edit_scripts_(s1, s2):
    differ = difflib.Differ()

    # no need to compare
    if len(s1) == 1:
        yield Replace(source=s1, target=s2)
        return

    # diff edits
    acts = [(act, char) for (act, _, char) in differ.compare(s1, s2)]
    i = 0
    while i < len(acts):
        a, c = acts[i]

        if a == ' ':
            # if the are only additions left, we should merge here
            if acts[i+1:] and acts[i+1][0] == '+':
                all_adds, cs = True, ''
                for (a, cc) in acts[i+1:]:
                    if a == '+':
                        cs += cc
                    elif a == '-':
                        all_adds = False
                        break
                    else:
                        break
                if all_adds:
                    yield Insert(string=cs, pos='post')
                    i += len(cs) + 1
                else:
                    yield Keep()
                    i += 1
            else:
                yield Keep()
                i += 1

        else:
            # grab edits involving any number of -/+ followed by any number of the other edit operation
            merge_a, merge_b = [(a, c)], []
            flipped = False
            for j in range(i + 1, len(acts)):
                a2, c2 = acts[j]
                if a2 == ' ':
                    break
                elif a2 == a:
                    if flipped:
                        break
                    else:
                        merge_a.append((a2, c2))
                else:
                    flipped = True
                    merge_b.append((a2, c2))

            # put always the substract first
            if merge_a[0][0] == '-':
                deletes, adds = merge_a, merge_b
            else:
                deletes, adds = merge_b, merge_a
            del merge_a, merge_b

            # do the actual merging
            if len(deletes) > len(adds):
                # zip and delete the remaining items
                for (_, c_b), (_, c_a) in zip(deletes, adds):
                    yield Replace(source=c_b, target=c_a)
                for (a, c) in deletes[len(adds):]:
                    yield Delete(string=c)

            elif len(adds) > len(deletes):
                # zip as many as possible and merge the remaining additions
                if deletes:
                    for idx in range(len(deletes) - 1):
                        (_, c_a), (_, c_b) = adds[idx], deletes[idx]
                        yield Replace(source=c_b, target=c_a)
                    _, c_b = deletes[-1]
                    c_a = ''.join([c for _, c in adds[len(deletes)-1:]])
                    yield Replace(source=c_b, target=c_a)
                else:
                    # this must mean that the additions are followed by a keep
                    if acts[i+len(adds)][0] == ' ':
                        c_a = ''.join([c for _, c in adds])
                        yield Insert(string=c_a, pos="pre")
                        i += 1
                    else:
                        raise ValueError("Expected a bunch of additions followed by keep")
            else:
                # same size, just zip them into replacements
                for (_, c_b), (_, c_a) in zip(deletes, adds):
                    yield Replace(source=c_b, target=c_a)

            i += len(adds) + len(deletes)


def transform(lem, tok):
    return tuple(get_edit_scripts_(tok, lem))


def inverse_transform(pred, tok):
    output, idx = "", 0
    for rule in pred:
        if isinstance(rule, Keep):
            output += tok[idx]
            idx += 1
        elif isinstance(rule, Replace):
            output += rule.target
            idx += len(rule.source)
        elif isinstance(rule, Insert):
            if rule.pos == 'pre':
                output += rule.string + tok[idx]
            else:
                output += tok[idx] + rule.string
            idx += 1
        else:
            idx += len(rule.string)
    return output
