
import os
import yaml
import json
from json_minify import json_minify


DEFAULTPATH = os.sep.join([os.path.dirname(__file__), 'default_settings.json'])


class Settings(dict):
    def __init__(self, *args, **kwargs):
        super(Settings, self).__init__(*args, **kwargs)
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
        super(Settings, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(Settings, self).__delitem__(key)
        del self.__dict__[key]


def flat_merge(s1, s2):
    """
    Merge two dictionaries in a flat way (non-recursive)

    >>> flat_merge({"a": {"b": 1}}, {"a": {"c": 2}})
    {'a': {'b': 1}}
    """
    for k in s2:
        if k not in s1:
            s1[k] = s2[k]

    return s1


def recursive_merge(s1, s2):
    """
    Recursively merge two dictionaries

    >>> recursive_merge({"a": {"b": 1}}, {"a": {"c": 2}})
    {'a': {'b': 1, 'c': 2}}
    """
    for k, v in s2.items():
        if k in s1 and isinstance(v, dict):
            if not isinstance(s1[k], dict):
                raise ValueError("Expected dictionary at key [{}]".format(k))
            s1[k] = recursive_merge(s1[k], v)
        elif k not in s1:
            s1[k] = v

    return s1


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

    # settings = Settings(flat_merge(p, defaults))
    settings = Settings(recursive_merge(p, defaults))

    # ultimately overwrite settings from environ vars of the form PIE_{var}
    checked = []
    for k in settings:
        env_k = 'PIE_{}'.format(k.upper())
        if env_k in os.environ:
            # transform to target type and overwrite settings
            settings[k] = type(defaults[k])(os.environ[env_k])
            checked.append(env_k)
    for env_k in os.environ:
        if env_k.startswith('PIE_') and env_k not in checked:
            raise ValueError(
                "Environment variable '{}' didn't match. Aborting!".format(env_k))

    # store the config path too:
    settings.config_path = config_path

    if settings.verbose:
        print("\n::: Loaded Config :::\n")
        print(yaml.dump(dict(settings)))

    return merge_task_defaults(settings)
