"""Paramiko transport: run commands and retrieve files over direct SSH to a pod.

Direct SSH (exposed port 22 + public IP) is used rather than RunPod's basic SSH proxy
because the proxy does not support SFTP, which we need to pull result manifests back.
"""

import logging
import shlex
import time
from collections.abc import Callable
from pathlib import Path
from types import TracebackType
from typing import Any, Self

from backdoord.cloud.errors import CloudError, RemoteCommandError

logger = logging.getLogger(__name__)


def _paramiko() -> Any:
    """Import paramiko lazily, raising a friendly error if the extra is missing."""

    try:
        import paramiko
    except ImportError as exc:
        raise CloudError(
            "paramiko not installed. Install with: uv sync --extra cloud"
        ) from exc

    return paramiko


class RemoteSession:
    """An SSH session to a pod, supporting streamed command execution and SFTP retrieval."""

    def __init__(
        self,
        host: str,
        port: int,
        key_path: Path,
        *,
        username: str = "root",
        connect_timeout: int = 60,
    ) -> None:
        """Initialize the session parameters.

        Args:
            host: Public SSH host.
            port: Public SSH port.
            key_path: Path to the local private key whose public half was injected.
            username: SSH username (RunPod pods use ``root``).
            connect_timeout: Per-attempt TCP connect timeout in seconds.
        """

        self._host = host
        self._port = port
        self._key_path = key_path.expanduser()
        self._username = username
        self._connect_timeout = connect_timeout
        self._client: Any = None

    def connect(self, *, retries: int = 12, backoff_s: float = 5.0) -> None:
        """Open the SSH connection, retrying while sshd warms up.

        Args:
            retries: Number of connection attempts.
            backoff_s: Seconds to wait between attempts.

        Raises:
            CloudError: If the connection cannot be established after all retries.
        """

        paramiko = _paramiko()
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        last_exc: Exception | None = None

        for attempt in range(1, retries + 1):
            try:
                client.connect(
                    hostname=self._host,
                    port=self._port,
                    username=self._username,
                    key_filename=str(self._key_path),
                    timeout=self._connect_timeout,
                    banner_timeout=self._connect_timeout,
                )
                self._client = client
                logger.info("SSH connected to %s:%d", self._host, self._port)

                return
            except Exception as exc:
                last_exc = exc
                logger.info(
                    "SSH attempt %d/%d failed (%s); retrying in %.0fs",
                    attempt,
                    retries,
                    exc,
                    backoff_s,
                )
                time.sleep(backoff_s)

        raise CloudError(
            f"Could not SSH to {self._host}:{self._port} after {retries} attempts: {last_exc}"
        )

    def run_command(
        self, command: str, *, wall_time_s: int, on_line: Callable[[str], None]
    ) -> int:
        """Run a command under a hard wall-time cap, streaming output line by line.

        The command is wrapped in ``timeout <wall_time_s> bash -lc '<command>'`` so it
        self-kills even if the host process dies. Exit status 124 means the timeout fired.

        Args:
            command: The remote shell command.
            wall_time_s: Hard remote wall-time cap in seconds.
            on_line: Callback invoked with each output line (stdout and stderr merged).

        Returns:
            The command's exit status (0 on success).

        Raises:
            CloudError: If called before :meth:`connect`.
            RemoteCommandError: If the command exits non-zero.
        """

        if self._client is None:
            raise CloudError("run_command called before connect()")

        wrapped = f"timeout {wall_time_s} bash -lc {shlex.quote(command)}"
        _, stdout, _ = self._client.exec_command(wrapped, get_pty=True)
        channel = stdout.channel

        buffer = ""
        while True:
            chunk = stdout.read(4096).decode("utf-8", errors="replace")

            if not chunk:
                break

            buffer += chunk
            *lines, buffer = buffer.split("\n")

            for line in lines:
                on_line(line)

        if buffer:
            on_line(buffer)

        status = channel.recv_exit_status()

        if status != 0:
            raise RemoteCommandError(command, status)

        return status

    def retrieve(self, remote_path: str, local_path: Path) -> None:
        """Download a remote file to a local path via SFTP.

        Args:
            remote_path: Absolute path on the pod.
            local_path: Local destination; parent directories are created.

        Raises:
            CloudError: If called before :meth:`connect`.
        """

        if self._client is None:
            raise CloudError("retrieve called before connect()")

        local_path.parent.mkdir(parents=True, exist_ok=True)
        sftp = self._client.open_sftp()

        try:
            sftp.get(remote_path, str(local_path))
            logger.info("Retrieved %s -> %s", remote_path, local_path)
        finally:
            sftp.close()

    def put_text(self, content: str, remote_path: str, *, mode: int = 0o600) -> None:
        """Write text to a remote file via SFTP (default mode 600, for secrets).

        Args:
            content: File contents to write.
            remote_path: Absolute destination path on the pod.
            mode: POSIX permission bits for the created file.

        Raises:
            CloudError: If called before :meth:`connect`.
        """

        if self._client is None:
            raise CloudError("put_text called before connect()")

        sftp = self._client.open_sftp()

        try:
            with sftp.file(remote_path, "w") as handle:
                handle.write(content)

            sftp.chmod(remote_path, mode)
        finally:
            sftp.close()

    def close(self) -> None:
        """Close the SSH connection if open."""

        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> Self:
        """Enter the context manager."""

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Close the session on context exit."""

        self.close()
