"""
Step 9.8: Structured Non-Sensitive Production Logging Configuration
"""

import time
import logging
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

# Configure standard logger format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

logger = logging.getLogger("hospital_queue_api")


class StructuredLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        start_time = time.time()
        response = await call_next(request)
        latency_ms = round((time.time() - start_time) * 1000, 2)

        # Non-sensitive operational log
        logger.info(
            f"method={request.method} path={request.url.path} "
            f"status={response.status_code} latency_ms={latency_ms}ms"
        )
        return response
