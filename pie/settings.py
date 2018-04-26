
import os
import json


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


def settings_from_file(config_path, verbose=True):
    """Loads and parses a parameter file.

    Parameters
    ===========
    config_path : str
        The path to the parameter file, formatted as json.

    Returns
    ===========
    settings : dict
        * A dictionary with the parameters
    """

    try:
        with open(config_path, 'r') as f:
            p = json.load(f)
    except Exception as e:
        raise ValueError(
            "Couldn't read config file: %s. Exception: %s" % (config_path, str(e)))

    settings = Settings(p)
    # add default values for missing settings:
    with open(os.sep.join((os.path.dirname(__file__),
                          'default_settings.json')), 'r') as f:
        defaults = json.load(f)

    for k in defaults:
        if k not in settings:
            settings[k] = defaults[k]

    # store the config path too:
    settings.config_path = config_path

    if verbose:
        print("::: Loaded Config :::")
        for k, v in settings.items():
            print("\t{} : {}".format(k, v))

    return settings
