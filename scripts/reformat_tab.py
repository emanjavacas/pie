
import os
import collections


ORIGINAL_TASK_ORDER = ('lemma', 'pos', 'morph')
EMPTY = '_'


def parse_morph(line):
    output = {}
    for field in line.split('|'):
        try:
            k, v = field.split('=')
            output[k] = v
        except ValueError:
            pass
    return output


def get_all_tasks(dirpath):
    all_tasks = collections.defaultdict(collections.Counter)
    for f in os.listdir(dirpath):
        with open(os.path.join(dirpath, f)) as f:
            try:
                for line in f:
                    _, *tasks = line.strip().split('\t')
                    if len(tasks) == 3:  # file has morphology
                        for task, value in parse_morph(tasks[2]).items():
                            all_tasks[task][value] += 1
            except UnicodeDecodeError:
                print("unicode")

    return all_tasks


def to_normal_format(f, output, all_tasks, sep='\t'):
    header = ['token', 'lemma', 'pos'] + all_tasks
    with open(f) as f, open(output, 'w+') as output:
        output.write(sep.join(header) + '\n')
        for line in f:
            token, *tasks = line.strip().split(sep)
            lemma, pos, morph = tasks
            morph = parse_morph(morph)
            line = [token, lemma, pos, *[morph.get(task, EMPTY) for task in all_tasks]]
            output.write(sep.join(line) + '\n')

if __name__ == '__main__':
    dirpath = 'datasets/capitula_classic/'

    all_tasks = get_all_tasks(dirpath)
    all_tasks = list(all_tasks)

    print(all_tasks)

    for f in os.listdir(dirpath):
        print("Processing file: ", f)
        to_normal_format(os.path.join(dirpath, f), f+'.csv', all_tasks)

    
