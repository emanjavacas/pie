#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Note: To use the 'upload' functionality of this file, you must:
#   $ pip install twine

import io
import os
import sys
from shutil import rmtree

from setuptools import find_packages, setup, Command

# Package meta-data.
NAME = 'pie'
DESCRIPTION = 'A Framework for Joint Learning of Sequence Labeling Tasks'
URL = 'https://github.com/emanjavacas/pie'
EMAIL = 'me@example.com'
AUTHOR = ' Enrique Manjavacas; Mike Kestemont'
REQUIRES_PYTHON = '>=3.6.0'
VERSION = "0.2.0a1"

# What packages are required for this module to be executed?
REQUIRED = [
    "lxml==4.2.1",
    "JSON_minify==0.3.0",
    "gensim==3.4.0",
    "tqdm==4.23.3",
    "numpy==1.14.3",
    "termcolor==1.1.0",
    "scikit_learn==0.19.1",
    "terminaltables==3.1.0",
    "torch>=1.0.1,<=1.2.0",
    "pyyaml @ https://github.com/yaml/pyyaml/archive/ccc40f3e2ba384858c0d32263ac3e3a6626ab15e.zip",
    "typing==3.6.6",
    "click==7.0"
]

# What packages are optional?
EXTRAS = {
    # 'fancy feature': ['django'],
    'webapp': [
        "flask==1.0.2",
        "gunicorn==19.9.0"
    ]
}

# The rest you shouldn't have to touch too much :)
# ------------------------------------------------
# Except, perhaps the License and Trove Classifiers!
# If you do change the License, remember to change the Trove Classifier for that!

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Load the package's __version__.py module as a dictionary.
about = {}
if not VERSION:
    project_slug = NAME.lower().replace("-", "_").replace(" ", "_")
    with open(os.path.join(here, project_slug, '__version__.py')) as f:
        exec(f.read(), about)
else:
    about['__version__'] = VERSION


class UploadCommand(Command):
    """Support setup.py upload."""

    description = 'Build and publish the package.'
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print('\033[1m{0}\033[0m'.format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status('Removing previous builds…')
            rmtree(os.path.join(here, 'dist'))
        except OSError:
            pass

        # Adds the commit for the verification when checking against
        #   the model git sha
        self.status('Writing commit SHA to pie/commit_build.py')
        COMMIT_BUILD = os.path.join(here, "pie", "commit_build.py")

        try:
            from pie.utils import GitInfo
            __commit__ = GitInfo(os.path.join(here, "pie", "__init__.py")).get_commit()
            with open(COMMIT_BUILD, "w") as f:
                f.write("COMMIT = '" + __commit__ + "'")
        except Exception as E:
            raise E

        self.status('Building Source and Wheel (universal) distribution…')
        os.system('{0} setup.py sdist bdist_wheel --universal'.format(sys.executable))

        self.status('Uploading the package to PyPI via Twine…')
        os.system('twine upload dist/*')

        self.status('Pushing git tags…')
        os.system('git tag v{0}'.format(about['__version__']))
        os.system('git push --tags')

        self.status('Reseting pie/commit_build.py')
        try:
            with open(COMMIT_BUILD, "w") as f:
                f.write("# DO NOT CHANGE THIS MANUALLY.\nCOMMIT = None")
        except Exception as E:
            raise E

        sys.exit()


# Where the magic happens:
setup(
    name=NAME,
    version=about['__version__'],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=('tests',)),
    # If your package is a single module, use this instead of 'packages':
    # py_modules=['mypackage'],

    entry_points={
        'console_scripts': ['pie=pie.scripts.group:pie_cli'],
    },
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    license='MIT',
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy',
        'Topic :: Text Processing :: Linguistic'
    ],
    # $ setup.py publish support.
    cmdclass={
        'upload': UploadCommand,
    },
)