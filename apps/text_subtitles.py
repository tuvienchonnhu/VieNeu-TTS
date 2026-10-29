"""Create downloadable SRT subtitles for generated speech."""
from __future__ import annotations

import math

from vieneu_utils.core_utils import (
    pack_sentences_into_chunks,
    split_into_sentences,
    split_text_into_paragraphs,
)


def _format_timestamp(milliseconds: int) -> str:
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, milliseconds = divmod(remainder, 1_000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"


def build_estimated_srt(text: str, duration_seconds: float, max_chars: int = 84) -> str:
    """Build sentence-sized captions with timestamps estimated from text length."""
    captions = []
    for paragraph in split_text_into_paragraphs(text):
        sentences = split_into_sentences(paragraph)
        for sentence in sentences:
            captions.extend(pack_sentences_into_chunks([sentence], max_chars=max_chars))

    if not captions:
        raise ValueError("Không có nội dung để tạo phụ đề.")
    if not math.isfinite(duration_seconds) or duration_seconds <= 0:
        raise ValueError("Thời lượng audio phải là số hữu hạn lớn hơn 0.")

    weights = [max(1, len(caption.strip())) for caption in captions]
    total_weight = sum(weights)
    duration_ms = max(round(duration_seconds * 1_000), len(captions))

    blocks = []
    elapsed_weight = 0
    start_ms = 0
    for index, (caption, weight) in enumerate(zip(captions, weights), start=1):
        elapsed_weight += weight
        end_ms = (
            duration_ms
            if index == len(captions)
            else round(duration_ms * elapsed_weight / total_weight)
        )
        end_ms = min(max(end_ms, start_ms + 1), duration_ms - (len(captions) - index))
        blocks.append(
            f"{index}\n"
            f"{_format_timestamp(start_ms)} --> {_format_timestamp(end_ms)}\n"
            f"{caption.strip()}"
        )
        start_ms = end_ms

    return "\n\n".join(blocks) + "\n"
