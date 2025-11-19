"""Tests for clean_filters module."""


from clean_filters import clean_text, remove_headers_footers


class TestCleanText:
    """Test cases for clean_text function."""

    def test_clean_text_empty(self):
        """Test that empty text returns empty string."""
        assert clean_text("") == ""
        assert clean_text(None) == ""

    def test_clean_text_removes_excessive_whitespace(self):
        """Test that excessive whitespace is removed."""
        text = "Hello\n\n\n\nWorld"
        result = clean_text(text)
        assert "\n\n\n" not in result

    def test_clean_text_removes_page_numbers(self):
        """Test that page numbers are removed."""
        text = "Some text\n5\nMore text"
        result = clean_text(text)
        assert result == "Some text\nMore text"

    def test_clean_text_removes_excessive_spaces(self):
        """Test that excessive spaces are removed."""
        text = "Hello    World"
        result = clean_text(text)
        assert result == "Hello World"

    def test_clean_text_with_extra_patterns(self):
        """Test that extra patterns are removed."""
        text = "Hello CONFIDENTIAL World"
        result = clean_text(text, extra_patterns=["CONFIDENTIAL"])
        assert "CONFIDENTIAL" not in result
        assert "Hello" in result
        assert "World" in result

    def test_clean_text_case_insensitive_patterns(self):
        """Test that pattern removal is case insensitive."""
        text = "Hello confidential World"
        result = clean_text(text, extra_patterns=["CONFIDENTIAL"])
        assert "confidential" not in result.lower()


class TestRemoveHeadersFooters:
    """Test cases for remove_headers_footers function."""

    def test_remove_headers(self):
        """Test that headers are removed."""
        text = "HEADER TEXT\nActual content here"
        result = remove_headers_footers(text, header_patterns=[r"HEADER TEXT"])
        assert "HEADER" not in result
        assert "Actual content" in result

    def test_remove_footers(self):
        """Test that footers are removed."""
        text = "Actual content here\nFOOTER TEXT"
        result = remove_headers_footers(text, footer_patterns=[r"FOOTER TEXT"])
        assert "FOOTER" not in result
        assert "Actual content" in result

    def test_no_patterns(self):
        """Test that text is unchanged when no patterns provided."""
        text = "Some text here"
        result = remove_headers_footers(text)
        assert result == text
