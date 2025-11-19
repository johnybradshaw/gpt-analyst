"""Pytest configuration and fixtures."""

import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_text():
    """Provide sample text for testing."""
    return """
    This is a sample document.

    It has multiple paragraphs and lines.

    Some content here with    excessive    spaces.

    1

    More content after a page number.
    """


@pytest.fixture
def mock_pdf_path(temp_dir):
    """Create a mock PDF file path."""
    pdf_path = temp_dir / "test.pdf"
    # Note: This creates a path but not an actual PDF file
    # For real PDF testing, you would need to create actual PDF files
    return pdf_path
