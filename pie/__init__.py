
from .utils import GitInfo

try:
    __commit__ = GitInfo(__file__).get_commit()
except Exception:
    from .commit_build import COMMIT
    if COMMIT:
        __commit__ = COMMIT
    else:
        import logging
        logging.warning(
            """
It seems like you downloaded `pie` instead of git-cloning it or installing it with pip.
We won't be able to check compatibility between pretrained models and `pie` version.
""")
        __commit__ = None

from . import utils
from . import trainer
from . import settings
from . import tagger
from . import initialization
from . import optimize
from .data import *
from .models import *
from .pretrain_encoder import Encoder
