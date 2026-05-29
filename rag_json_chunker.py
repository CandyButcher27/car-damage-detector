from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass
class Chunk:
    source_file: str
    document_type: str
    chunk_id: str
    text: str
    metadata: dict[str, Any]


def _iter_json_files(inputs: list[str]) -> Iterable[Path]:
    seen: set[Path] = set()
    for item in inputs:
        path = Path(item)
        if path.is_dir():
            for candidate in sorted(path.rglob("*.json")):
                if candidate not in seen:
                    seen.add(candidate)
                    yield candidate
        elif path.is_file() and path.suffix.lower() == ".json":
            if path not in seen:
                seen.add(path)
                yield path


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return " ".join(value.split())
    return " ".join(str(value).split())


def _flatten_value(value: Any, prefix: str = "") -> list[str]:
    lines: list[str] = []
    if isinstance(value, dict):
        for key, subvalue in value.items():
            next_prefix = f"{prefix}{key}" if not prefix else f"{prefix}.{key}"
            lines.extend(_flatten_value(subvalue, next_prefix))
    elif isinstance(value, list):
        if not value:
            lines.append(f"{prefix}: []")
        else:
            for index, item in enumerate(value, start=1):
                item_prefix = f"{prefix}[{index}]" if prefix else f"[{index}]"
                lines.extend(_flatten_value(item, item_prefix))
    else:
        text = _normalize_text(value)
        if prefix:
            lines.append(f"{prefix}: {text}")
        elif text:
            lines.append(text)
    return lines


def _looks_like_ocr_pages(obj: Any) -> bool:
    return isinstance(obj, dict) and isinstance(obj.get("pages"), list)


def _make_text_chunks(lines: list[str], *, max_chars: int, overlap_lines: int) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    def flush() -> None:
        nonlocal current, current_len
        text = "\n".join(current).strip()
        if text:
            chunks.append(text)
        if overlap_lines > 0 and current:
            current = current[-overlap_lines:]
            current_len = sum(len(line) + 1 for line in current)
        else:
            current = []
            current_len = 0

    for line in lines:
        if not line:
            continue
        line_len = len(line) + 1
        if current and current_len + line_len > max_chars:
            flush()
        current.append(line)
        current_len += line_len

    flush()
    return chunks


def _chunk_ocr_document(path: Path, payload: dict[str, Any], max_chars: int, overlap_lines: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    pages = payload.get("pages") or []
    source_name = str(path)

    for page_index, page in enumerate(pages, start=1):
        page_number = page.get("page", page_index) if isinstance(page, dict) else page_index
        lines_obj = page.get("lines", []) if isinstance(page, dict) else []
        lines: list[str] = []

        for entry in lines_obj:
            if isinstance(entry, dict):
                text = _normalize_text(entry.get("text", ""))
            else:
                text = _normalize_text(entry)
            if text:
                lines.append(text)

        if not lines:
            continue

        page_chunks = _make_text_chunks(lines, max_chars=max_chars, overlap_lines=overlap_lines)
        for chunk_index, text in enumerate(page_chunks, start=1):
            chunks.append(
                Chunk(
                    source_file=source_name,
                    document_type="ocr_pages",
                    chunk_id=f"page-{page_number}-chunk-{chunk_index}",
                    text=text,
                    metadata={
                        "page": page_number,
                        "chunk_index": chunk_index,
                        "source_type": "ocr_pages",
                    },
                )
            )

    return chunks


def _chunk_structured_document(path: Path, payload: Any, max_chars: int) -> list[Chunk]:
    source_name = str(path)
    lines = _flatten_value(payload)
    chunks: list[Chunk] = []

    if not lines:
        return chunks

    text_chunks = _make_text_chunks(lines, max_chars=max_chars, overlap_lines=0)
    document_type = "structured_json"

    for chunk_index, text in enumerate(text_chunks, start=1):
        chunks.append(
            Chunk(
                source_file=source_name,
                document_type=document_type,
                chunk_id=f"chunk-{chunk_index}",
                text=text,
                metadata={
                    "chunk_index": chunk_index,
                    "source_type": document_type,
                },
            )
        )

    return chunks


def chunk_json_file(path: Path, max_chars: int, overlap_lines: int) -> list[Chunk]:
    payload = _load_json(path)
    if _looks_like_ocr_pages(payload):
        return _chunk_ocr_document(path, payload, max_chars=max_chars, overlap_lines=overlap_lines)
    return _chunk_structured_document(path, payload, max_chars=max_chars)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Chunk OCR/structured JSON files into embedding-ready JSONL for RAG"
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="One or more JSON files or directories containing JSON files",
    )
    parser.add_argument(
        "--output",
        default="rag_chunks.jsonl",
        help="Output JSONL file (default: rag_chunks.jsonl)",
    )
    parser.add_argument(
        "--max_chars",
        type=int,
        default=1200,
        help="Maximum characters per chunk before splitting (default: 1200)",
    )
    parser.add_argument(
        "--overlap_lines",
        type=int,
        default=3,
        help="Overlap lines between OCR page chunks (default: 3)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_files = list(_iter_json_files(args.inputs))
    if not input_files:
        raise SystemExit("No JSON files found in the provided inputs.")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_chunks = 0
    with output_path.open("w", encoding="utf-8") as f:
        for json_path in input_files:
            try:
                chunks = chunk_json_file(
                    json_path,
                    max_chars=args.max_chars,
                    overlap_lines=args.overlap_lines,
                )
            except Exception as exc:
                raise SystemExit(f"Failed to process {json_path}: {exc}") from exc

            for chunk in chunks:
                record = {
                    "source_file": chunk.source_file,
                    "document_type": chunk.document_type,
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "metadata": chunk.metadata,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_chunks += 1

    print(f"Wrote {total_chunks} chunk(s) to {output_path}")


if __name__ == "__main__":
    main()