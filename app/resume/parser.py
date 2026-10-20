"""Lightweight resume parsing pipeline."""

from __future__ import annotations

import logging
import asyncio
from pathlib import Path

from app.models.resume import ResumeParseMetadata, ResumeParseResult
from app.resume.links import extract_resume_links
from app.resume.normalizer import normalize_resume_text
from app.errors import document_error

logger = logging.getLogger(__name__)

VISION_OCR_PROMPT = (
    "This is a resume image. Extract all visible text faithfully. Preserve section "
    "headers, bullet points, project names, dates, links, and contact information. "
    "Do not summarize. Output the extracted text in the original language."
)


async def parse_pdf(file_path: str | Path) -> tuple[str, ResumeParseMetadata]:
    """Extract text from a PDF resume using pdfplumber."""
    return await asyncio.to_thread(_parse_pdf_sync, file_path)


def _parse_pdf_sync(file_path: str | Path) -> tuple[str, ResumeParseMetadata]:
    import pdfplumber

    file_path = Path(file_path)
    text_parts: list[str] = []

    with pdfplumber.open(str(file_path)) as pdf:
        page_count = len(pdf.pages)
        for index, page in enumerate(pdf.pages):
            page_text = page.extract_text() or ""
            if page_text:
                text_parts.append(page_text)
            logger.debug("PDF page %d: %d chars", index + 1, len(page_text))

    raw_text = "\n\n".join(text_parts)
    metadata = ResumeParseMetadata(
        file_name=file_path.name,
        file_type=file_path.suffix.lower(),
        parser="pdfplumber",
        page_count=page_count,
        raw_char_count=len(raw_text),
    )

    if not raw_text.strip():
        raise ValueError(
            "PDF contains no extractable text. It may be scanned or image-based."
        )

    logger.info("PDF parsed: %d pages, %d chars", page_count, len(raw_text))
    return raw_text, metadata


async def parse_image_with_vision(file_path: str | Path) -> tuple[str, ResumeParseMetadata]:
    """Extract resume text from an image using the configured vision LLM."""
    from app.llm.client import get_llm_client

    file_path = Path(file_path)
    llm = get_llm_client()
    logger.info("Parsing image resume via vision LLM: %s", file_path.name)

    raw_text = await llm.chat_vision(image_path=file_path, prompt=VISION_OCR_PROMPT)
    if not raw_text.strip():
        raise document_error("EMPTY_DOCUMENT", "图片中未识别到可用的简历文本。", 422)

    metadata = ResumeParseMetadata(
        file_name=file_path.name,
        file_type=file_path.suffix.lower(),
        parser="vision_ocr",
        raw_char_count=len(raw_text),
    )
    logger.info("Vision OCR extracted %d chars from %s", len(raw_text), file_path.name)
    return raw_text, metadata


async def parse_resume_document(file_path: str | Path) -> ResumeParseResult:
    """Parse a resume file into normalized text, links, metadata, and warnings."""
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()
    warnings: list[str] = []

    if suffix == ".pdf":
        try:
            raw_text, metadata = await parse_pdf(file_path)
        except ValueError as exc:
            raise document_error(
                "EMPTY_DOCUMENT",
                "该 PDF 没有可提取文本，可能是扫描件；请改为上传 PNG 或 JPG 图片。",
                422,
            ) from exc
        except Exception as exc:
            raise document_error(
                "INVALID_DOCUMENT", "PDF 文件损坏或无法解析，请检查文件后重试。", 422
            ) from exc
    elif suffix in (".png", ".jpg", ".jpeg"):
        raw_text, metadata = await parse_image_with_vision(file_path)
    else:
        raise document_error(
            "UNSUPPORTED_FILE_TYPE",
            "不支持该文件类型，请上传 PDF、PNG、JPG 或 JPEG。",
            415,
        )

    normalized_text, normalize_warnings = normalize_resume_text(raw_text)
    warnings.extend(normalize_warnings)
    links = extract_resume_links(normalized_text)

    metadata.normalized_char_count = len(normalized_text)
    if not links:
        warnings.append("No links or contact items were detected in the resume text.")

    return ResumeParseResult(
        raw_text=raw_text,
        normalized_text=normalized_text,
        links=links,
        metadata=metadata,
        warnings=warnings,
    )


async def parse_resume_file(file_path: str | Path) -> str:
    """Compatibility helper returning only normalized resume text."""
    result = await parse_resume_document(file_path)
    return result.normalized_text
