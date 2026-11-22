# Settings read from environment variables (.env). Every other file imports from here.
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Format is "<provider>:<model id>"; init_chat_model can't guess the provider without the prefix.
AGENT_MODEL = os.getenv("AGENT_MODEL", "groq:openai/gpt-oss-120b")

DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
DATASETS_DIR = DATA_DIR / "datasets"
RUNS_DIR = DATA_DIR / "runs"

MAX_INPUT_CHARS = int(os.getenv("MAX_INPUT_CHARS", "1000"))
MAX_DOWNLOAD_MB = int(os.getenv("MAX_DOWNLOAD_MB", "200"))
MAX_ROWS = int(os.getenv("MAX_ROWS", "50000"))
MIN_ROWS = int(os.getenv("MIN_ROWS", "50"))
RECURSION_LIMIT = int(os.getenv("RECURSION_LIMIT", "25"))

# Tools that pause the graph until a human approves them.
SENSITIVE_TOOLS = {"download_dataset", "train_baselines"}

# LangSmith only traces when told to, so switch it on whenever a key is configured.
if os.getenv("LANGSMITH_API_KEY"):
    os.environ.setdefault("LANGSMITH_TRACING", "true")

DATASETS_DIR.mkdir(parents=True, exist_ok=True)
RUNS_DIR.mkdir(parents=True, exist_ok=True)
