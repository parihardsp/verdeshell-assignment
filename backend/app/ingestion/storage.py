"""Session-based local file storage manager.

Stores uploaded files on the local machine under `data/uploads/{session_id}_{date}/`.
"""

import re
import uuid
from datetime import UTC, datetime
from pathlib import Path

from fastapi import UploadFile

from app.core.config import get_settings

_settings = get_settings()


class SessionFileStorageService:
    """Manages disk persistence of uploaded documents partitioned by session and date."""

    def __init__(self, root_upload_dir: str | None = None):
        self.root_dir = Path(root_upload_dir or _settings.UPLOAD_DIR).resolve()
        self.root_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def generate_user_id() -> str:
        """Create a standard unique user identifier (e.g. 'usr_1001')."""
        from app.core.logging import generate_standard_user_id

        return generate_standard_user_id()

    @staticmethod
    def generate_session_id() -> str:
        """Create a standard unique session identifier (e.g. 'sess_1001')."""
        from app.core.logging import generate_standard_session_id

        return generate_standard_session_id()

    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Strip directory traversal paths and disallowed characters."""
        base_name = Path(filename).name
        cleaned = re.sub(r"[^\w\.\- ]", "_", base_name)
        return cleaned or "uploaded_file"

    def get_session_directory(
        self, session_id: str, user_id: str | None = None
    ) -> tuple[Path, str]:
        """Resolve and ensure the storage directory for a session/user and current date.

        Returns:
            Tuple of (Path to directory, date string YYYY-MM-DD)
        """
        date_str = datetime.now(UTC).strftime("%Y-%m-%d")
        uid_part = f"{user_id}_" if user_id else ""
        session_folder_name = f"{uid_part}{session_id}_{date_str}"
        session_path = self.root_dir / session_folder_name
        session_path.mkdir(parents=True, exist_ok=True)
        return session_path, date_str

    async def save_uploaded_file(
        self, upload_file: UploadFile, session_id: str, user_id: str | None = None
    ) -> tuple[Path, int]:
        """Save an incoming FastAPI UploadFile to the session's disk directory.

        Returns:
            Tuple of (Saved absolute Path, size in bytes)
        """
        session_dir, _ = self.get_session_directory(session_id, user_id=user_id)
        clean_name = self.sanitize_filename(upload_file.filename or "file")
        target_path = session_dir / clean_name

        if target_path.exists():
            stem = target_path.stem
            suffix = target_path.suffix
            target_path = session_dir / f"{stem}_{uuid.uuid4().hex[:4]}{suffix}"

        bytes_written = 0
        with open(target_path, "wb") as f:
            while chunk := await upload_file.read(1024 * 64):  # 64KB buffers
                f.write(chunk)
                bytes_written += len(chunk)

        await upload_file.seek(0)
        return target_path, bytes_written
