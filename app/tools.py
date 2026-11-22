# Tools the agent can call. Each validates its arguments and returns a string,
# including errors, so the LLM can read what went wrong and recover.
import pandas as pd
from langchain_core.tools import tool

from app import guardrails as g
from app import kaggle_client, ml
from app.config import DATASETS_DIR


def dataset_folder(ref: str):
    return DATASETS_DIR / ref.replace("/", "__")


def load_dataset_file(dataset_ref: str, file_name: str) -> pd.DataFrame:
    """Load one file of a downloaded dataset, refusing paths outside its folder."""
    folder = dataset_folder(g.check_dataset_ref(dataset_ref))
    path = g.resolve_inside(folder, file_name)
    if not path.is_file():
        raise g.GuardrailViolation(f"File '{file_name}' not found in {dataset_ref}. Download it first?")
    return ml.load_table(path)


@tool
def search_datasets(query: str) -> str:
    """Search Kaggle for datasets and list up to 5 candidates.

    Args:
        query: Keywords, e.g. "house prices".
    """
    try:
        datasets = kaggle_client.search(query)
    except Exception as error:
        return f"Error searching Kaggle: {error}"
    if not datasets:
        return f"No datasets found for '{query}'."
    lines = [f"- ref={d.ref} | {d.title} | {d.vote_count} votes | license={d.license_name}" for d in datasets]
    return "Candidates:\n" + "\n".join(lines)


@tool
def download_dataset(dataset_ref: str) -> str:
    """Download a Kaggle dataset. A human must approve this first.

    Args:
        dataset_ref: Ref from the search results, formatted "owner/dataset-slug".
    """
    try:
        ref = g.check_dataset_ref(dataset_ref)
        dataset = kaggle_client.find(ref)
        if dataset is None:
            return f"Error: dataset '{ref}' not found."
        g.check_download_size(dataset.total_bytes)
        folder = dataset_folder(ref)
        kaggle_client.download(ref, folder)
    except g.GuardrailViolation as error:
        return f"Blocked by guardrail: {error}"
    except Exception as error:
        return f"Error downloading '{dataset_ref}': {error}"

    files = [p.relative_to(folder) for p in folder.rglob("*") if p.suffix.lower() in g.TABULAR_EXTENSIONS]
    if not files:
        return f"Downloaded {ref}, but it has no tabular files."
    return f"Downloaded {ref}. Tabular files:\n" + "\n".join(f"- {f}" for f in sorted(files))


@tool
def inspect_file(dataset_ref: str, file_name: str) -> str:
    """Show the shape, columns, missing values and sample rows of a downloaded file.

    Args:
        dataset_ref: The downloaded dataset, "owner/dataset-slug".
        file_name: A tabular file listed after download, e.g. "train.csv".
    """
    try:
        df = load_dataset_file(dataset_ref, file_name)
    except g.GuardrailViolation as error:
        return f"Blocked by guardrail: {error}"
    except Exception as error:
        return f"Error reading '{file_name}': {error}"

    columns = pd.DataFrame({"dtype": df.dtypes.astype(str), "missing_%": (df.isna().mean() * 100).round(1), "unique": df.nunique()})
    return f"{len(df)} rows x {df.shape[1]} columns\n{columns.to_string()}\n\nSample rows:\n{df.head(3).to_string()}"


@tool
def train_baselines(dataset_ref: str, file_name: str, target_column: str, task_type: str = "auto") -> str:
    """Train baseline models (dummy, linear, random forest) and report metrics. A human must approve this first.

    Args:
        dataset_ref: The downloaded dataset, "owner/dataset-slug".
        file_name: The tabular file to train on.
        target_column: The column to predict, exactly as named in inspect_file output.
        task_type: "classification", "regression" or "auto".
    """
    try:
        df = load_dataset_file(dataset_ref, file_name)
        results, pipelines = ml.train_baselines(df, target_column, task_type)
        run_id = ml.save_run(results, pipelines, dataset_ref, target_column)
    except g.GuardrailViolation as error:
        return f"Blocked by guardrail: {error}"
    except Exception as error:
        return f"Error training models: {error}"

    table = pd.DataFrame(results).drop(columns=["task_type", "score"]).to_string(index=False)
    best = max(results, key=lambda r: r["score"])["model"]
    return f"Task: {results[0]['task_type']}, target: {target_column}\n{table}\nBest model: {best}. Run id: {run_id}"


TOOLS = [search_datasets, download_dataset, inspect_file, train_baselines]
