"""Errors serialized the way Chroma's server does, so clients raise the same exceptions.

The `error` field must be a key of `chromadb.errors.error_types`.
"""

from __future__ import annotations


class ChromaError(Exception):
    status_code = 500
    error_name = "ChromaError"

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message

    def to_dict(self) -> dict:
        return {"error": self.error_name, "message": self.message}


class InvalidArgumentError(ChromaError):
    status_code = 400
    error_name = "InvalidArgumentError"


class NotFoundError(ChromaError):
    status_code = 404
    error_name = "NotFoundError"


class DuplicateError(ChromaError):
    """Chroma 1.5 answers duplicate tenant/database/collection names with 409 + ChromaError."""

    status_code = 409
    error_name = "ChromaError"


class AuthError(ChromaError):
    status_code = 401
    error_name = "AuthError"


class InternalError(ChromaError):
    status_code = 500
    error_name = "InternalError"


class NotSupportedError(ChromaError):
    status_code = 501
    error_name = "ChromaError"
