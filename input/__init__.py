"""
Input Processing Package
Provides document ingest and cleaning for text, PDF, and DOCX resumes.
"""

from input.file_processor import (
    clean_text,
    extract_from_docx,
    extract_from_pdf,
    extract_from_txt,
    read_uploaded_file,
)

__all__ = [
    "clean_text",
    "extract_from_pdf",
    "extract_from_docx",
    "extract_from_txt",
    "read_uploaded_file",
]
