"""Pure, review-oriented helpers for splitting extracted PDF text.

The functions in this module deliberately stop before approval or publishing.
They only suggest structure and possible monetary figures so that a human can
compare each chunk with the source PDF.  In particular, an amount candidate is
not a fact and a classified section is not verified evidence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


_WHITESPACE_RE = re.compile(r"\s+")
_AMOUNT_RE = re.compile(r"(?<!\d)\d[\d,]*(?:\.\d+)?\s*บาท")

# Explicit ``ข้อ`` markers are the least ambiguous and may be flattened into
# the middle of one OCR line.  Decimal and parenthesised markers cover the
# numbered rows used in the grant tables.  The filters in ``_find_markers``
# reject common monetary/reference uses before any split is made.
_MARKER_RE = re.compile(
    r"(?P<thai>(?<!\S)ข้อ\s*(?:ที่\s*)?\d+(?:\s*\.\s*\d+)*)"
    r"|(?P<decimal>(?<![\S\d.])\d{1,2}(?:\s*\.\s*\d{1,2})+(?![\d.])(?=\s|$))"
    r"|(?P<paren>(?<!\S)\(\s*\d{1,2}(?:\s*\.\s*\d{1,2})*\s*\)(?=\s|$))"
)

_REFERENCE_PREFIXES = {
    "กล่าวใน",
    "จาก",
    "ดู",
    "ตาม",
    "ตามที่",
    "ภายใต้",
    "ยกเว้น",
    "อ้าง",
    "อ้างอิง",
    "อ้างถึง",
    "ใน",
}
_NON_MARKER_SUFFIXES = {
    "%",
    "คน",
    "ครั้ง",
    "ทุน",
    "บาท",
    "ปี",
    "เดือน",
    "วัน",
    "เปอร์เซ็นต์",
}

_SECTION_LABELS = {
    "title": "ชื่อเรื่อง",
    "preamble": "บทนำและที่มา",
    "definition": "นิยาม",
    "fund_list": "รายการทุน",
    "amount_table": "วงเงินและอัตรา",
    "eligibility": "คุณสมบัติและสิทธิ์",
    "application": "การสมัครและเอกสาร",
    "timeline": "กำหนดเวลา",
    "contact": "ข้อมูลติดต่อ",
    "other": "ข้อความอื่น",
}


def section_label_for(section_type: str) -> str:
    """Return the canonical Thai label for one supported draft section type."""
    try:
        return _SECTION_LABELS[section_type]
    except KeyError as exc:
        raise ValueError("unsupported section type") from exc


@dataclass(frozen=True)
class _Marker:
    start: int
    end: int
    label: str
    kind: str
    parent: str | None = None


def _normalise_space(value: str) -> str:
    """Return stable single-spaced text without changing visible characters."""

    return _WHITESPACE_RE.sub(" ", value.replace("\u00a0", " ")).strip()


def _amount_candidates(text: str, limit: int = 25) -> list[str]:
    """Return de-duplicated monetary strings as *review hints* only."""

    return list(
        dict.fromkeys(match.group().strip() for match in _AMOUNT_RE.finditer(text))
    )[:limit]


def classify_section(clause: str, text: str) -> dict:
    """Suggest a conservative section type for a prospective chunk.

    Classification is intentionally based on strong wording found in Thai
    administrative documents.  Ambiguous prose stays ``other`` rather than
    being promoted into a fact/rule category.  The return value contains UI
    metadata only; it carries no approval state.
    """

    clause_text = _normalise_space(clause or "")
    body = _normalise_space(text or "")
    lowered = body.lower()

    section_type = "other"

    # Definitions take precedence because their examples may also mention
    # applications, eligibility, or monetary units without establishing any
    # operational rule about those topics.
    if re.search(r"(?:หมายความว่า|หมายถึง|ให้ความหมายว่า|คำนิยาม)", body):
        section_type = "definition"
    elif re.search(
        r"(?:ยื่น(?:แบบ)?คำขอ|ยื่นขอรับทุน|แบบคำขอ|เอกสารประกอบ|"
        r"หลักฐานประกอบ|วิธีการสมัคร|ขั้นตอนการ(?:สมัคร|ยื่น)|สมัครผ่าน)",
        body,
    ):
        section_type = "application"
    elif re.search(
        r"(?:กำหนดการ|ระยะเวลา(?:รับสมัคร|ดำเนินการ)|วัน(?:เปิด|ปิด)รับสมัคร|"
        r"เปิดรับสมัคร|ปิดรับสมัคร|ภายในวันที่|ไม่เกินวันที่)",
        body,
    ):
        section_type = "timeline"
    elif re.search(
        r"(?:ติดต่อ(?:สอบถาม)?|โทรศัพท์|โทร\.?|อีเมล|e-?mail|เว็บไซต์|website)",
        lowered,
    ):
        section_type = "contact"
    elif re.search(
        r"(?:คุณสมบัติ(?:ของ)?ผู้ขอ|ผู้ขอรับทุนต้อง|ผู้มีสิทธิ(?:ขอ|รับ)|"
        r"มีสิทธิ(?:ขอ|รับ)|ไม่มีสิทธิ|เงื่อนไขการรับทุน|"
        r"ต้องเป็น(?:บุคลากร|ผู้ปฏิบัติงาน)|สามารถขอรับทุน)",
        body,
    ):
        section_type = "eligibility"
    elif _AMOUNT_RE.search(body) or re.search(
        r"(?:วงเงินสูงสุด|วงเงินรวม|อัตราการ(?:จ่าย|สนับสนุน)|"
        r"งบประมาณต่อทุน|จำนวนเงิน(?:สนับสนุน)?|เพดานวงเงิน)",
        body,
    ):
        section_type = "amount_table"
    elif re.search(
        r"(?:ประเภทของทุน|ประเภททุน|รายการทุน|ทุน(?:ส่งเสริม|สนับสนุน).{0,30}ประกอบด้วย|"
        r"แบ่งทุนออกเป็น)",
        body,
    ):
        section_type = "fund_list"
    elif (
        (body.startswith("ประกาศ") or body.startswith("หลักเกณฑ์"))
        and len(body) <= 240
        and clause_text in {"", "ทั้งหน้า"}
    ):
        section_type = "title"
    elif (
        re.match(r"^(?:ด้วย|โดยที่|ตามที่|อาศัยอำนาจ)", body)
        and clause_text in {"", "ทั้งหน้า"}
    ):
        section_type = "preamble"

    return {
        "section_type": section_type,
        "section_label": section_label_for(section_type),
    }


def _previous_word(text: str, position: int) -> str:
    prefix = text[:position].rstrip()
    if not prefix:
        return ""
    match = re.search(r"([^\s,;:()]+)$", prefix)
    return match.group(1) if match else ""


def _next_word(text: str, position: int) -> str:
    suffix = text[position:].lstrip()
    match = re.match(r"([^\s,;:()]+)", suffix)
    return match.group(1) if match else ""


def _normalise_marker(raw: str, kind: str) -> str:
    raw = _normalise_space(raw)
    if kind == "thai":
        number = re.search(r"\d+(?:\s*\.\s*\d+)*", raw)
        assert number is not None
        return "ข้อ " + re.sub(r"\s*\.\s*", ".", number.group())
    if kind == "decimal":
        return re.sub(r"\s*\.\s*", ".", raw)
    number = re.sub(r"\s*\.\s*", ".", raw.strip()[1:-1].strip())
    return f"({number})"


def _numeric_parts(label: str) -> list[str]:
    return re.findall(r"\d+", label)


def _derived_parent(label: str, kind: str) -> str | None:
    parts = _numeric_parts(label)
    if len(parts) < 2:
        return None
    parent_number = ".".join(parts[:-1])
    if kind == "thai":
        return f"ข้อ {parent_number}"
    if kind == "paren":
        return f"({parent_number})"
    return parent_number


def _find_markers(text: str) -> list[_Marker]:
    markers: list[_Marker] = []
    current_major: str | None = None
    parent_aliases: dict[str, str] = {}
    suppressed_reference_end: int | None = None

    for match in _MARKER_RE.finditer(text):
        kind = match.lastgroup or ""
        raw = match.group()

        # If a cited ``ข้อ 4`` was rejected, an immediately attached ``(1.5)``
        # belongs to the same citation and must not become an orphaned chunk.
        if suppressed_reference_end is not None:
            if not text[suppressed_reference_end : match.start()].strip():
                suppressed_reference_end = match.end()
                continue
            suppressed_reference_end = None

        previous = _previous_word(text, match.start()).lower()
        following = _next_word(text, match.end()).lower()

        # ``ตาม ข้อ 2`` and ``จำนวน 10.5 บาท`` are references/amounts, not
        # structural boundaries.  Explicit markers at the beginning are kept.
        if match.start() and any(previous.endswith(prefix) for prefix in _REFERENCE_PREFIXES):
            suppressed_reference_end = match.end()
            continue
        if kind in {"decimal", "paren"} and following in _NON_MARKER_SUFFIXES:
            continue

        label = _normalise_marker(raw, kind)
        parent = _derived_parent(label, kind)
        if kind == "paren" and parent is None:
            parent = current_major
        if parent in parent_aliases:
            parent = parent_aliases[parent]

        marker = _Marker(match.start(), match.end(), label, kind, parent)

        # Source citations and table rows often use ``ข้อ 4 (1.5)`` as one
        # marker.  Keep that compound label together instead of emitting an
        # empty ``ข้อ 4`` chunk followed by an orphaned parenthesised chunk.
        if (
            markers
            and kind == "paren"
            and not text[markers[-1].end : marker.start].strip()
            and markers[-1].kind == "thai"
        ):
            major = markers.pop()
            local_label = marker.label
            compound_label = f"{major.label} {local_label}"
            marker = _Marker(
                major.start,
                marker.end,
                compound_label,
                "compound",
                major.label,
            )
            # Later markers such as ``(1.1)`` derive ``(1)`` as their
            # structural parent.  Point that local label at the actual
            # compound chunk rather than at a label that was never emitted.
            parent_aliases[local_label] = compound_label

        markers.append(marker)
        if kind == "thai":
            current_major = label
            numeric_label = ".".join(_numeric_parts(label))
            if numeric_label:
                parent_aliases[numeric_label] = label

    return markers


def _preferred_break(text: str, start: int, hard_end: int) -> int:
    """Find a readable break near ``hard_end`` without exceeding it."""

    if hard_end >= len(text):
        return len(text)

    minimum = start + max(1, int((hard_end - start) * 0.65))
    window = text[minimum:hard_end]
    candidates = [
        window.rfind(" "),
        window.rfind(";"),
        window.rfind(":"),
        window.rfind("।"),
    ]
    relative = max(candidates)
    return minimum + relative + 1 if relative >= 0 else hard_end


def _split_long_section(text: str, max_chars: int) -> list[str]:
    """Hard-cap one section, with small overlap only when it was too long."""

    if len(text) <= max_chars:
        return [text]

    # Ten percent (at most 72 characters) gives a reviewer enough boundary
    # context without duplicating substantial portions of a section.
    overlap = min(72, max_chars // 10)
    parts: list[str] = []
    start = 0

    while start < len(text):
        hard_end = min(len(text), start + max_chars)
        end = _preferred_break(text, start, hard_end)
        if end <= start:
            end = hard_end
        part = text[start:end].strip()
        if part:
            # Stripping can only shorten a part, so the cap remains absolute.
            parts.append(part[:max_chars])
        if end >= len(text):
            break

        next_start = max(start + 1, end - overlap)
        if next_start < len(text) and next_start > 0:
            # Prefer beginning the overlap at a word boundary when one is
            # nearby.  Thai OCR often has no spaces, in which case hard slicing
            # is deterministic and still lossless.
            boundary = text.find(" ", next_start, min(end, next_start + 24))
            if boundary >= 0:
                next_start = boundary + 1
        start = next_start

    return parts


def _section_parent(marker: _Marker, last_major: str | None) -> str | None:
    if marker.parent:
        return marker.parent
    if marker.kind == "paren":
        return last_major
    return _derived_parent(marker.label, marker.kind)


def split_structured_page(
    text: str,
    page_number: int,
    extraction_method: str,
    max_chars: int = 900,
) -> list[dict]:
    """Split one extracted page into ordered, review-only chunk proposals.

    Numbered markers are recognised even when OCR flattened the whole page to
    one line.  Every returned ``text`` is non-empty and at most ``max_chars``.
    Short numbered clauses are retained; only empty whitespace is discarded.
    """

    if max_chars < 1:
        raise ValueError("max_chars must be at least 1")
    if page_number < 1:
        raise ValueError("page_number must be at least 1")

    source = (text or "").replace("\u00a0", " ").strip()
    if not source:
        return []

    markers = _find_markers(source)
    sections: list[tuple[str, str | None, str]] = []

    if not markers:
        sections.append(("ทั้งหน้า", None, _normalise_space(source)))
    else:
        preamble = _normalise_space(source[: markers[0].start])
        if preamble:
            sections.append(("ทั้งหน้า", None, preamble))

        last_major: str | None = None
        for index, marker in enumerate(markers):
            end = markers[index + 1].start if index + 1 < len(markers) else len(source)
            body = _normalise_space(source[marker.start:end])
            if not body:
                continue
            parent = _section_parent(marker, last_major)
            sections.append((marker.label, parent, body))
            if marker.kind == "thai":
                last_major = marker.label
            elif marker.kind == "compound":
                last_major = marker.parent

    chunks: list[dict] = []
    for clause, parent_clause, section_text in sections:
        # A long logical section keeps one type across all of its physical
        # parts.  Reclassifying continuation fragments would turn them into
        # ``other`` merely because the heading appeared in part zero.
        classification = classify_section(clause, section_text)
        for part_index, part in enumerate(_split_long_section(section_text, max_chars)):
            chunks.append(
                {
                    "text": part,
                    "page": page_number,
                    "extraction_method": extraction_method,
                    "order_in_page": len(chunks),
                    "clause": clause,
                    "parent_clause": parent_clause,
                    "part_index": part_index,
                    **classification,
                    "amount_candidates": _amount_candidates(part),
                }
            )

    return chunks
