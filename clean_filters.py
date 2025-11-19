"""
Text cleaning utilities for PDF extraction.

This module provides functions to clean and filter extracted text from PDFs.
"""

import re
from typing import List, Optional


def clean_text(text: str, extra_patterns: Optional[List[str]] = None) -> str:
    """
    Clean extracted text by removing common artifacts and unwanted patterns.

    Args:
        text: The raw text to clean
        extra_patterns: Optional list of additional patterns to remove (case-insensitive)

    Returns:
        Cleaned text with artifacts removed
    """
    if not text:
        return ""

    # Remove common PDF artifacts
    # Remove excessive whitespace
    text = re.sub(r"\n\s*\n\s*\n", "\n\n", text)

    # Remove page numbers (common patterns)
    text = re.sub(r"\n\s*\d+\s*\n", "\n", text)

    # Remove isolated single characters
    text = re.sub(r"\n\s*[a-zA-Z]\s*\n", "\n", text)

    # Remove excessive spaces
    text = re.sub(r" +", " ", text)

    # Remove common header/footer artifacts
    text = re.sub(r"(\n|^)([A-Z\s]{2,})\s*\d+\s*$", "", text, flags=re.MULTILINE)

    # Apply extra patterns if provided
    if extra_patterns:
        for pattern in extra_patterns:
            if pattern:
                # Case-insensitive removal of extra patterns
                text = re.sub(re.escape(pattern), "", text, flags=re.IGNORECASE)

    # Final cleanup: normalize whitespace
    text = re.sub(r"\n\n\n+", "\n\n", text)
    text = text.strip()

    return text


def remove_headers_footers(
    text: str,
    header_patterns: Optional[List[str]] = None,
    footer_patterns: Optional[List[str]] = None,
) -> str:
    """
    Remove common headers and footers from text.

    Args:
        text: The text to clean
        header_patterns: List of regex patterns for headers
        footer_patterns: List of regex patterns for footers

    Returns:
        Text with headers and footers removed
    """
    if header_patterns:
        for pattern in header_patterns:
            text = re.sub(pattern, "", text, flags=re.MULTILINE | re.IGNORECASE)

    if footer_patterns:
        for pattern in footer_patterns:
            text = re.sub(pattern, "", text, flags=re.MULTILINE | re.IGNORECASE)

    return text
