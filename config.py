from __future__ import annotations
import logging, logging.config, os
from zoneinfo import ZoneInfo
import structlog
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from structlog.typing import FilteringBoundLogger

SERVICE_NAME = os.getenv("SERVICE_NAME", "lpbot")
GIT_SHA = os.getenv("GIT_SHA")
SERVICE_VERSION = os.getenv("SERVICE_VERSION") or GIT_SHA or "dev"

class Settings(BaseSettings):
    TELEGRAM_BOT_TOKEN: str | None = None
    TELEGRAM_CHAT_ID: int | None = None
    TELEGRAM_ENABLED: bool = True

    METEORA_PAIR_ADDRESS: str
    METEORA_BASE_URL: str = "https://dlmm-api.meteora.ag"

    CHECK_EVERY_MINUTES: int = 15
    COOLDOWN_MINUTES: int = 60
    DB_PATH: str = "./app.db"
    HTTP_UA_MAIN: str = "outtaband/0.1 (+https://github.com/desclei/OuttaBand)"
    HTTP_UA_VOL: str = "outtaband-volatility/0.1 (+https://github.com/dsesclei/OuttaBand)"

    BAND_A: str | None = None
    BAND_B: str | None = None
    BAND_C: str | None = None

    BINANCE_BASE_URL: str = "https://api.binance.us"
    BINANCE_SYMBOL: str = "SOLUSDT"
    VOL_CACHE_TTL_SECONDS: int = 60
    VOL_MAX_STALE_SECONDS: int = 7200

    LOCAL_TZ: str = "America/New_York"
    DAILY_LOCAL_HOUR: int = 8
    DAILY_LOCAL_MINUTE: int = 0
    DAILY_ENABLED: bool = True
    INCLUDE_A_ON_HIGH: bool = False

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra="ignore")

    @model_validator(mode="after")
    def _validate(self) -> "Settings":
        from zoneinfo import ZoneInfo as _ZI
        if not (0 <= self.DAILY_LOCAL_HOUR <= 23): raise ValueError("DAILY_LOCAL_HOUR must be between 0 and 23 inclusive")
        if not (0 <= self.DAILY_LOCAL_MINUTE <= 59): raise ValueError("DAILY_LOCAL_MINUTE must be between 0 and 59 inclusive")
        if self.VOL_CACHE_TTL_SECONDS < 5: raise ValueError("VOL_CACHE_TTL_SECONDS must be at least 5 seconds")
        if self.VOL_MAX_STALE_SECONDS < self.VOL_CACHE_TTL_SECONDS: raise ValueError("VOL_MAX_STALE_SECONDS must be >= VOL_CACHE_TTL_SECONDS")
        if self.COOLDOWN_MINUTES < 1: raise ValueError("COOLDOWN_MINUTES must be at least 1")
        if self.CHECK_EVERY_MINUTES < 1: raise ValueError("CHECK_EVERY_MINUTES must be at least 1")
        try: _ZI(self.LOCAL_TZ)
        except Exception as exc: raise ValueError(f"Invalid LOCAL_TZ '{self.LOCAL_TZ}'") from exc
        if self.TELEGRAM_ENABLED and (not self.TELEGRAM_BOT_TOKEN or self.TELEGRAM_CHAT_ID is None):
            raise ValueError("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID are required when TELEGRAM_ENABLED=true")
        return self

def configure_logging() -> FilteringBoundLogger:
    level = os.getenv("LPBOT_LOG_LEVEL", "INFO").upper()
    level_num = {
        "CRITICAL": logging.CRITICAL, "ERROR": logging.ERROR, "WARNING": logging.WARNING,
        "WARN": logging.WARNING, "INFO": logging.INFO, "DEBUG": logging.DEBUG, "NOTSET": logging.NOTSET,
    }.get(level, logging.INFO)

    timestamper = structlog.processors.TimeStamper(fmt="iso", key="ts", utc=True)
    logging.config.dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "structlog": {"()": structlog.stdlib.ProcessorFormatter,
                          "processors": [structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                                         structlog.processors.JSONRenderer()],
                          "foreign_pre_chain": [structlog.processors.add_log_level, timestamper]}
        },
        "handlers": {"default": {"class": "logging.StreamHandler", "formatter": "structlog", "stream": "ext://sys.stdout"}},
        "loggers": {"": {"handlers": ["default"], "level": level_num, "propagate": True}},
    })
    structlog.configure(
        processors=[structlog.contextvars.merge_contextvars, structlog.stdlib.add_logger_name,
                    structlog.stdlib.add_log_level, timestamper, structlog.stdlib.PositionalArgumentsFormatter(),
                    structlog.processors.format_exc_info, structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        context_class=dict, logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.make_filtering_bound_logger(level_num), cache_logger_on_first_use=True,
    )
    base = structlog.get_logger("lpbot").bind(service=SERVICE_NAME, version=SERVICE_VERSION)
    return base.bind(git_sha=GIT_SHA) if GIT_SHA else base

def service_meta() -> dict[str, str | None]:
    return {"service": SERVICE_NAME, "version": SERVICE_VERSION, "git_sha": GIT_SHA}

def tz(settings: Settings) -> ZoneInfo:
    return ZoneInfo(settings.LOCAL_TZ)
