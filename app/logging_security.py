"""Remove query strings from server access/handshake logs without logging secrets."""
import logging
import re


class QueryStringFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno <= logging.DEBUG:
            # WebSocket protocol DEBUG records contain raw authentication frames.
            return False
        # Uvicorn HTTP uses positional arguments; WS handshake messages use a
        # different shape. Sanitizing the rendered message covers both forms.
        message = record.getMessage()
        record.msg = re.sub(r"\?[^\s\"'<>]*", "?[redacted]", message)
        record.args = ()
        return True


class AccessQueryFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        # Preserve the five fields required by Uvicorn's AccessFormatter.
        if isinstance(record.args, tuple) and len(record.args) == 5:
            args = list(record.args)
            args[2] = str(args[2]).split("?", 1)[0]
            record.args = tuple(args)
        else:
            QueryStringFilter().filter(record)
        return True


def install_safe_access_logging() -> None:
    logging.getLogger("websockets").setLevel(logging.WARNING)
    for name, filter_type in (("uvicorn.access", AccessQueryFilter), ("uvicorn.error", QueryStringFilter)):
        logger = logging.getLogger(name)
        if not any(isinstance(item, filter_type) for item in logger.filters):
            logger.addFilter(filter_type())
