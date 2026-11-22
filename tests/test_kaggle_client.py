from types import SimpleNamespace

from app import kaggle_client


def test_every_built_client_gets_a_user_agent():
    """Guards the workaround for the SDK bug where User-Agent was None."""

    def build_client():  # mimics the SDK: a fresh client each call, user agent unset
        return SimpleNamespace(_http_client=SimpleNamespace(_user_agent=None))

    api = SimpleNamespace(build_kaggle_client=build_client)
    kaggle_client._add_missing_user_agent(api)

    for _ in range(2):  # a new client is built per API call, so check more than once
        assert api.build_kaggle_client()._http_client._user_agent
