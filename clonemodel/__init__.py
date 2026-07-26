import warnings
from importlib.metadata import PackageNotFoundError, version

warnings.filterwarnings("ignore", module="torchaudio")
warnings.filterwarnings(
    "ignore",
    category=SyntaxWarning,
    message="invalid escape sequence",
    module="pydub.utils",
)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="torch.distributed.algorithms.ddp_comm_hooks",
)

try:
    __version__ = version("mimicvoice")
except PackageNotFoundError:
    __version__ = "0.0.0"

try:
    from clonemodel.models.mimicvoice import (
        MimicVoice,
        MimicVoiceConfig,
        MimicVoiceGenerationConfig,
    )
    __all__ = ["MimicVoice", "MimicVoiceConfig", "MimicVoiceGenerationConfig"]
except ImportError:
    __all__ = []
