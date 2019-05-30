
import os
import yaml
import json
from datetime import datetime
from json_minify import json_minify

from pie import utils


DEFAULTPATH = os.sep.join([os.path.dirname(__file__), 'default_settings.json'])


class SettingsDict(dict):
    def __init__(self, *args, **kwargs):
        super(SettingsDict, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v

        if kwargs:
            for k, v in kwargs.iteritems():
                self[k] = v

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(SettingsDict, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(SettingsDict, self).__delitem__(key)
        del self.__dict__[key]


class Settings(SettingsDict):
    """
    Subclass to hold settings specific functions defined over default settings
    """
    def get_targets(self):
        return [task['name'] for task in self.tasks if task.get('target')]

    def get_fname_infix(self):
        # fname
        fname = os.path.join(self.modelpath, self.modelname)
        timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        infix = '+'.join(self.get_targets()) + '-' + timestamp
        return fname, infix


def merge_task_defaults(settings):
    for task in settings.tasks:
        task_settings = task.get("settings", {})
        task_settings["target"] = task_settings.get("target", task['name'])
        task['settings'] = task_settings
        for tkey, tval in settings.task_defaults.items():
            task[tkey] = task.get(tkey, tval)

    return settings


def load_default_settings():
    """
    Load built-in default settings
    """
    with open(DEFAULTPATH) as f:
        return merge_task_defaults(Settings(json.loads(json_minify(f.read()))))


def parse_key(keys, v, defaults):
    """
    >>> parse_key(['a', 'b'], '1', {'a': {'b': 2}})
    {'a': {'b': 1}}
    """
    key, *keys = keys
    if key not in defaults:
        raise ValueError("Unknown key: ", key)
    if not keys:
        return {key: type(defaults[key])(v)}
    return {key: parse_key(keys, v, defaults[key])}


def parse_env_settings(defaults):
    output = {}
    for k, v in os.environ.items():
        if not k.startswith('PIE_'):
            continue
        keys = k.lower()[4:].split('__')
        output = utils.recursive_merge(output, parse_key(keys, v, defaults))

    return output


def check_settings(settings):
    # - check at least and at most one target
    has_target = False
    for task in settings.tasks:
        if len(settings.tasks) == 1:
            task['target'] = True
        if task.get('target', False):
            if has_target:
                raise ValueError("Got more than one target task")
            has_target = True
    if not has_target:
        raise ValueError("Needs at least one target task")

    return settings


def settings_from_file(config_path):
    """Loads and parses a parameter file.

    Parameters
    ===========
    config_path : str
        The path to the parameter file, formatted as json.

    Returns
    ===========
    settings : dict, A dictionary with the parameters
    """

    try:
        with open(config_path, 'r') as f:
            p = json.loads(json_minify(f.read()))
    except Exception as e:
        raise ValueError(
            "Couldn't read config file: %s. Exception: %s" % (config_path, str(e)))

    # add default values for missing settings:
    with open(DEFAULTPATH, 'r') as f:
        defaults = json.loads(json_minify(f.read()))

    settings = Settings(
        utils.recursive_merge(
            # merge defaults
            utils.recursive_merge(p, defaults),
            # ultimately overwrite settings from environ vars of the form PIE_{var}
            parse_env_settings(defaults), overwrite=True))

    # store the config path too:
    settings.config_path = config_path

    if settings.verbose:
        print("\n::: Loaded Config :::\n")
        print(yaml.dump(dict(settings)))

    return check_settings(merge_task_defaults(settings))
