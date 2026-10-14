"""Runtime settings: Streamlit secrets first, then environment variables.

Hosted deployments (Streamlit Community Cloud, Hugging Face Spaces) provide secrets through
`st.secrets` (or as environment variables); locally a `.env` file is loaded into the environment.
`get_setting` checks `st.secrets` and falls back to the environment, so the same code runs everywhere.
"""

import os


def _streamlit_secret(name: str):
    """The secret `name`, or None if Streamlit isn't available or has no such secret / no secrets file."""
    try:
        import streamlit as st

        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:  # no secrets.toml, not running under Streamlit, ...
        pass
    return None


def get_setting(name: str, default=None):
    value = _streamlit_secret(name)
    if value is None or value == "":
        value = os.environ.get(name)
    return value if value not in (None, "") else default


def get_int_setting(name: str, default: int = 0) -> int:
    try:
        return int(get_setting(name, default))
    except (TypeError, ValueError):
        return default
