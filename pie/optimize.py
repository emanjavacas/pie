
import random
import json

import yaml
from json_minify import json_minify
import scipy.stats as stats

from pie import utils
from pie.settings import settings_from_file, check_settings, merge_task_defaults
from pie.settings import Settings


# available distributions
class truncnorm:
    def __init__(self, mu, std, lower=0, upper=1):
        a, b = (lower - mu) / std, (upper - mu) / std
        self.norm = stats.truncnorm(a, b, mu, std)

    def rvs(self):
        return float(self.norm.rvs())


class normint:
    def __init__(self, mu, std, lower, upper):
        self.norm = truncnorm(mu, std, lower, upper)

    def rvs(self):
        return int(round(self.norm.rvs())) // 2 * 2


class choice:
    def __init__(self, items):
        self.items = items

    def rvs(self):
        return random.choice(self.items)


def parse_opt(obj, opt_key):
    """
    Parses the opt file into a (possibly deep) dictionary where the leaves are 
    ready-to-use distributions
    """
    opt = {}

    for param, v in obj.items():
        if isinstance(v, list):
            opt[param] = [parse_opt(v_item, opt_key) for v_item in v]
        elif isinstance(v, dict):
            if opt_key in v:
                if v[opt_key] == 'norm':
                    opt[param] = stats.norm(**v['params'])
                elif v[opt_key] == 'truncnorm':
                    opt[param] = truncnorm(**v['params'])
                elif v[opt_key] == 'normint':
                    opt[param] = normint(**v['params'])
                elif v[opt_key] == 'choice':
                    opt[param] = choice(v['params'])
                else:
                    raise ValueError("Unknown distribution: ", v[opt_key])
            else:
                opt[param] = parse_opt(v, opt_key)
        else:
            opt[param] = v

    return opt


def read_opt(path, opt_key='opt'):
    """
    Reads and parses the opt file (as per parse_opt)
    """
    with open(path) as f:
        obj = json.loads(json_minify(f.read()))

    return parse_opt(obj, opt_key)


def sample_from_config(opt):
    """
    Applies the distributions specified in the opt.json file
    """
    output = {}

    for param, dist in opt.items():
        if isinstance(dist, dict):
            output[param] = sample_from_config(dist)
        elif isinstance(dist, list):
            output[param] = [sample_from_config(d) for d in dist]
        elif isinstance(dist, (str, float, int, bool)):
            output[param] = dist  # no sampling
        else:
            output[param] = dist.rvs()

    return output


def run_optimize(train_fn, config, opt, n_iter, **kwargs):
    for i in range(n_iter):
        print()
        print("::: Starting optimization run {} :::".format(i + 1))
        print()
        sampled = sample_from_config(opt)
        merged = Settings(
            utils.recursive_merge(dict(config), sampled, overwrite=True))
        print("::: Sampled config :::")
        print(yaml.dump(dict(merged)))
        train_fn(check_settings(merge_task_defaults(merged)), **kwargs)
