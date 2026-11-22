# Thin wrapper around the Kaggle API: search and download only.
from pathlib import Path

_api = None


def get_api():
    """Create and authenticate the Kaggle client once. Imported lazily because
    `import kaggle` authenticates immediately and would break tests without credentials."""
    global _api
    if _api is None:
        from kaggle.api.kaggle_api_extended import KaggleApi

        _api = KaggleApi()
        _api.authenticate()
        _add_missing_user_agent(_api)
    return _api


def _add_missing_user_agent(api) -> None:
    """Workaround for a bug in kaggle 1.8.2 / kagglesdk 0.1.13: every client it builds has User-Agent=None,
    which makes `requests` reject the call. The SDK builds a fresh client per call, so wrap the builder.
    Uses private SDK attributes; remove once a fixed kaggle release is available."""
    build_client = api.build_kaggle_client

    def build_client_with_user_agent():
        client = build_client()
        client._http_client._user_agent = "kaggle-api/v1.8.2"
        return client

    api.build_kaggle_client = build_client_with_user_agent


def search(query: str, limit: int = 5) -> list:
    """Return the most-voted datasets matching the query."""
    results = get_api().dataset_list(search=query, sort_by="votes") or []
    return [d for d in results if d][:limit]


def find(ref: str):
    """Look up one dataset by 'owner/slug'. Returns None if it doesn't exist."""
    # Searching the full "owner/slug" works; the SDK's `user=` filter returned nothing in testing.
    results = get_api().dataset_list(search=ref) or []
    return next((d for d in results if d and d.ref == ref), None)


def download(ref: str, folder: Path) -> None:
    """Download and unzip a dataset into folder."""
    get_api().dataset_download_files(ref, path=str(folder), unzip=True, quiet=True)
