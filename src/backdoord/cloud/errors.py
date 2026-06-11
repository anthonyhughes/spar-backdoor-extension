"""Exceptions for the RunPod cloud launcher."""


class CloudError(Exception):
    """Base error for cloud orchestration failures."""


class PreflightError(CloudError):
    """Raised when a safety or validation check fails before any pod is provisioned."""


class NoCapacityError(CloudError):
    """Raised when RunPod has no instances available for the requested GPU type / cloud tier."""


class PodTimeoutError(CloudError):
    """Raised when a pod does not become ready within the allotted time."""


class RemoteCommandError(CloudError):
    """Raised when a remote command exits non-zero (exit 124 indicates the wall-time timeout fired)."""

    def __init__(self, command: str, exit_status: int) -> None:
        """Store the failing command and its exit status."""

        self.command = command
        self.exit_status = exit_status
        super().__init__(f"Remote command exited {exit_status}: {command}")
