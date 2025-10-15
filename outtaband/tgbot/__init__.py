from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .app import TelegramApp

__all__ = ["TelegramApp"]


def __getattr__(name: str) -> object:
    if name == "TelegramApp":
        from .app import TelegramApp

        return TelegramApp
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
