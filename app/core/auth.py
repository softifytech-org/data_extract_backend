"""No authentication - explicit, deliberate decision.

This application has no accounts, no shared key, no registration, and no
access gate of any kind, on any endpoint. Every route is fully public to
anyone who can reach the URL. See app/api/batches.py's module docstring
and the final report for exactly what that means and what it risks
(unauthenticated uploads, OCR cost exposure, no isolation between
callers).

CurrentUser/get_current_user_optional are kept only as a stable, unused
stub so app/api/documents.py's existing Depends(get_current_user_optional)
parameters and _check_access(...) calls don't need to be removed from
every route individually - a smaller, lower-risk diff than rewriting
~10 function signatures for a function that would do nothing anyway.
get_current_user_optional never inspects the request and always returns
None.
"""
from __future__ import annotations


class CurrentUser:
    def __init__(self, user_id: str, email: str) -> None:
        self.user_id = user_id
        self.email = email


def get_current_user_optional() -> CurrentUser | None:
    return None
