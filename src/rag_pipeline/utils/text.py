"""Utility helpers for text normalization used across the pipeline."""

from __future__ import annotations

import re
from typing import Callable, Iterable, List

_NON_BREAKING_SPACE_PATTERN = re.compile(r"\u00A0")
_MULTISPACE_PATTERN = re.compile(r"[ \t]{2,}")
_MULTI_NEWLINE_PATTERN = re.compile(r"\n{3,}")
_TRAILING_SPACE_BEFORE_NEWLINE = re.compile(r"[ \t]+\n")


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace while preserving intentional newlines."""

    text = _NON_BREAKING_SPACE_PATTERN.sub(" ", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _TRAILING_SPACE_BEFORE_NEWLINE.sub("\n", text)
    text = _MULTISPACE_PATTERN.sub(" ", text)
    text = _MULTI_NEWLINE_PATTERN.sub("\n\n", text)
    return text.strip()


def strip_bom(text: str) -> str:
    """Remove UTF Byte Order Marks that may appear in CSV exports."""

    return text.lstrip("\ufeff")


def build_cleaning_pipeline(functions: Iterable[Callable[[str], str]]) -> List[Callable[[str], str]]:
    """Freeze an iterable of cleaning callables into a list."""

    return list(functions)


DEFAULT_CLEANING_PIPELINE = build_cleaning_pipeline([strip_bom, normalize_whitespace])


__all__ = [
    "normalize_whitespace",
    "strip_bom",
    "build_cleaning_pipeline",
    "DEFAULT_CLEANING_PIPELINE",
]
