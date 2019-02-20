

from .utils import GitInfo

try:
    from .commit_build import COMMIT
    __commit__ = COMMIT
except (ImportError, SyntaxError):
    try:
        __commit__ = GitInfo(__file__).get_commit()
    except Exception:
        import logging
        logging.warn(
            """
    It seems like you download `pie` instead of git-cloning it.
    We won't be able to check compatibility between pretrained models and `pie` version
            """)
        __commit__ = None

from . import utils
from . import trainer
from . import settings
from . import tagger
from . import initialization
from .data import *
from .models import *
from .pretrain_encoder import Encoder
