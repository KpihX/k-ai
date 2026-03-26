# src/k_ai/auth.py
"""
OAuth token loaders registry.

TOKEN_LOADERS maps oauth_provider_name → Callable[[token_path, scopes], str].
Each loader receives the token file path and requested scopes, and must return
a valid bearer token string (refreshing if needed).

To add a new OAuth provider:
    1. Implement your loader function.
    2. Register it in TOKEN_LOADERS below.
"""
from typing import Callable, Dict, List

# Signature: (token_path: str, scopes: List[str]) -> str
TokenLoader = Callable[[str, List[str]], str]

TOKEN_LOADERS: Dict[str, TokenLoader] = {
    # "google": _load_google_token,   # example — add loaders here
}
