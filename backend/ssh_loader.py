"""SSH-based remote dataset loader.

Connects via paramiko using either password or private-key auth, reads a
remote file, and saves it into backend/storage/datasets/.
"""
from __future__ import annotations

import io
import os
from pathlib import Path

import paramiko


def fetch_via_ssh(host: str, port: int, username: str,
                  password: str | None,
                  private_key: str | None,
                  remote_path: str,
                  local_path: str | Path) -> dict:
    """Returns {"size": int, "remote_path": str, "local_path": str}."""
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    transport = paramiko.Transport((host, int(port)))
    pkey = None
    if private_key:
        try:
            pkey = paramiko.RSAKey.from_private_key(io.StringIO(private_key))
        except Exception:
            try:
                pkey = paramiko.Ed25519Key.from_private_key(io.StringIO(private_key))
            except Exception:
                pkey = None
    if pkey is not None:
        transport.connect(username=username, pkey=pkey)
    else:
        transport.connect(username=username, password=password or "")
    try:
        sftp = paramiko.SFTPClient.from_transport(transport)
        sftp.get(remote_path, str(local_path))
        sftp.close()
    finally:
        transport.close()
    return {"size": local_path.stat().st_size,
            "remote_path": remote_path,
            "local_path": str(local_path)}
