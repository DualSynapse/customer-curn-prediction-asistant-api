import opik
from loguru import logger
from opik.integrations.langchain import OpikTracer
from src.config.settings import get_settings

settings = get_settings()

# Konfigurasi Opik sekali saat module di-import
try:
    opik.configure(
        api_key=settings.OPIK_API_KEY,
        url=settings.OPIK_URL,
    )
    logger.info("Opik configured successfully")
except Exception as e:
    logger.warning("Failed to configure Opik: {}", e)


def get_tracer() -> OpikTracer | None:
    """Buat instance OpikTracer baru per request. Return None jika Opik tidak tersedia."""
    try:
        return OpikTracer()
    except Exception as e:
        logger.warning("Failed to create OpikTracer: {}", e)
        return None