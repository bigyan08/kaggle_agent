# Plain-Python safety checks. They run outside the LLM, so the model can't talk its way past them.
import re
from pathlib import Path

import pandas as pd

from app.config import MAX_DOWNLOAD_MB, MAX_INPUT_CHARS, MIN_ROWS

TABULAR_EXTENSIONS = {".csv", ".tsv", ".xlsx", ".xls", ".json", ".parquet"}

# Catches only obvious prompt-injection attempts; easy to extend.
INJECTION_PATTERN = re.compile(
    r"ignore (all |any )?(previous|prior|above) instructions"
    r"|reveal .{0,20}(system prompt|api key)"
    r"|(\.env|kaggle\.json)",
    re.IGNORECASE,
)

# Kaggle dataset refs look like "owner/dataset-slug".
DATASET_REF_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}/[A-Za-z0-9][A-Za-z0-9_.-]{0,99}$")


class GuardrailViolation(ValueError):
    """Raised when input or a tool argument fails a safety check."""


def check_user_input(message: str) -> str:
    """Reject empty, oversized or injection-looking messages."""
    message = message.strip()
    if not message:
        raise GuardrailViolation("Message is empty.")
    if len(message) > MAX_INPUT_CHARS:
        raise GuardrailViolation(f"Message is longer than {MAX_INPUT_CHARS} characters.")
    if INJECTION_PATTERN.search(message):
        raise GuardrailViolation("Message looks like a prompt-injection attempt.")
    return message


def check_dataset_ref(ref: str) -> str:
    """Only allow refs shaped like 'owner/dataset'."""
    ref = ref.strip()
    if not DATASET_REF_PATTERN.match(ref):
        raise GuardrailViolation(f"'{ref}' is not a valid dataset ref (expected 'owner/dataset').")
    return ref


def check_download_size(total_bytes: int | None) -> None:
    """Refuse datasets above the size limit."""
    if total_bytes and total_bytes > MAX_DOWNLOAD_MB * 1024 * 1024:
        raise GuardrailViolation(f"Dataset is {total_bytes // 1024**2} MB, limit is {MAX_DOWNLOAD_MB} MB.")


def resolve_inside(base: Path, relative: str) -> Path:
    """Join base and relative, refusing paths that escape base (e.g. '../../etc')."""
    base = base.resolve()
    target = (base / relative).resolve()
    if not target.is_relative_to(base):
        raise GuardrailViolation(f"Path '{relative}' is outside the allowed folder.")
    return target


def check_target_column(df: pd.DataFrame, target_column: str) -> None:
    """The target must be a real column of the table."""
    if target_column not in df.columns:
        raise GuardrailViolation(f"Column '{target_column}' not found. Columns: {list(df.columns)}")


def check_enough_rows(row_count: int) -> None:
    """Too few rows gives meaningless metrics."""
    if row_count < MIN_ROWS:
        raise GuardrailViolation(f"Only {row_count} usable rows, need at least {MIN_ROWS}.")


def check_classes(target: pd.Series) -> None:
    """Classification needs at least 2 classes with 2+ rows each (required for a stratified split)."""
    counts = target.value_counts()
    if len(counts) < 2 or counts.min() < 2:
        raise GuardrailViolation("Classification needs at least 2 classes with 2+ rows each.")
