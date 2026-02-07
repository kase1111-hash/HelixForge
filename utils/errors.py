"""Custom exception hierarchy for HelixForge.

All application-specific exceptions inherit from HelixForgeError,
enabling structured error handling at API boundaries.
"""


class HelixForgeError(Exception):
    """Base exception for all HelixForge errors."""

    def __init__(self, message: str, detail: str = ""):
        self.message = message
        self.detail = detail
        super().__init__(message)


class ConfigurationError(HelixForgeError):
    """Invalid or missing configuration."""
    pass


class IngestionError(HelixForgeError):
    """Error during data ingestion."""
    pass


class InterpretationError(HelixForgeError):
    """Error during metadata interpretation."""
    pass


class AlignmentError(HelixForgeError):
    """Error during schema alignment."""
    pass


class FusionError(HelixForgeError):
    """Error during dataset fusion."""
    pass


class InsightError(HelixForgeError):
    """Error during statistical analysis."""
    pass
