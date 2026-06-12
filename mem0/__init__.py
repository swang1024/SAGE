import importlib.metadata

try:
    # This fork ships as the "sage" distribution (see pyproject.toml), registered
    # by `uv pip install -e .` / `pip install -e .`.
    __version__ = importlib.metadata.version("sage")
except importlib.metadata.PackageNotFoundError:
    # Imported straight from a clone without an editable/standard install
    # (the benchmark runner adds the repo root to sys.path). Do NOT look up
    # "mem0ai" here: this is a fork, that distribution is intentionally not installed.
    __version__ = "local"

from mem0.client.main import AsyncMemoryClient, MemoryClient  # noqa
from mem0.memory.main import AsyncMemory, Memory  # noqa
