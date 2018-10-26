
import difflib


def get_edit_scripts_(s1, s2):
    differ = difflib.Differ()

    # no need to compare
    if len(s1) == 1:
        yield '<replace string="{}" with="{}">'.format(s1, s2)
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
                    yield '<insert string="{}" post>'.format(cs)
                    i += len(cs) + 1
                else:
                    yield '<keep>'
                    i += 1
            else:
                yield '<keep>'
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
                    yield '<replace string="{}" with="{}">'.format(c_b, c_a)
                for (a, c) in deletes[len(adds):]:
                    yield '<delete string="{}">'.format(c)

            elif len(adds) > len(deletes):
                # zip as many as possible and merge the remaining additions
                if deletes:
                    for idx in range(len(deletes) - 1):
                        (_, c_a), (_, c_b) = adds[idx], deletes[idx]
                        yield '<replace string="{}" with="{}">'.format(c_b, c_a)
                    _, c_b = deletes[-1]
                    c_a = ''.join([c for _, c in adds[len(deletes)-1:]])
                    yield '<replace string="{}" with="{}">'.format(c_b, c_a)
                else:
                    # this must mean that the additions are followed by a keep
                    if acts[i+len(adds)][0] == ' ':
                        c_a = ''.join([c for _, c in adds])
                        yield '<insert string="{}" pre>'.format(c_a)
                        i += 1
                    else:
                        raise ValueError("Expected a bunch of additions followed by keep")
            else:
                # same size, just zip them into replacements
                for (_, c_b), (_, c_a) in zip(deletes, adds):
                    yield '<replace string="{}" with="{}">'.format(c_b, c_a)

            i += len(adds) + len(deletes)


def greedy_scripts(tok, lem):
    return tuple(get_edit_scripts_(tok, lem))