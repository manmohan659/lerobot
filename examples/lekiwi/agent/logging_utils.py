import json
import logging
import os
import socket
import sys
import time
import uuid
from typing import Any, Dict, Optional


def get_session_id() -> str:
    sid = os.environ.get("SESSION_ID")
    if sid:
        return sid
    sid = uuid.uuid4().hex[:12]
    os.environ["SESSION_ID"] = sid
    return sid


class JsonFormatter(logging.Formatter):
    def __init__(self, static_fields: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()
        self.static_fields = static_fields or {}

    def format(self, record: logging.LogRecord) -> str:  # type: ignore[override]
        payload: Dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)) + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "host": socket.gethostname(),
            **self.static_fields,
        }
        # Merge extra dict if record.msg was a dict
        if isinstance(record.args, dict):
            payload.update(record.args)
        return json.dumps(payload, ensure_ascii=False)


def setup_json_logger(name: str, level: int = logging.INFO, log_file: Optional[str] = None, **static: Any) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    formatter = JsonFormatter(static_fields=static)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    logger.propagate = False
    return logger


def log_event(logger: logging.Logger, event: str, **fields: Any) -> None:
    logger.info(event, extra={"event": event, **fields})


