"""Create subtitles aligned to the generated audio chunks."""
from __future__ import annotations

import logging
import os
import re

import numpy as np

from vieneu_utils.core_utils import (
    pause_pad_samples,
    split_into_sentences,
    split_text_into_paragraphs,
)

_WORD = re.compile(r"[^\W_]+", re.UNICODE)
_LOGGER = logging.getLogger(__name__)


def _report_alignment_warning(message: str, warnings: list[str] | None) -> None:
    if warnings is None:
        _LOGGER.warning(message)
    else:
        warnings.append(message)


def _format_timestamp(milliseconds: int) -> str:
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, milliseconds = divmod(remainder, 1_000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"


def _captions_for_chunk(text: str) -> list[str]:
    captions = []
    for paragraph in split_text_into_paragraphs(text):
        captions.extend(split_into_sentences(paragraph))
    return [caption.strip() for caption in captions if caption.strip()] or [text.strip()]


def _source_chunks(source_text: str, normalized_chunks: list[str]) -> list[str]:
    """Recover exact source spelling for each chunk using its normalized words."""
    source_words = list(_WORD.finditer(source_text))
    source_cursor = 0
    source_char_cursor = 0
    result = []

    for chunk in normalized_chunks:
        word_cursor = source_cursor
        matched = []
        for word in _WORD.finditer(chunk):
            target = word.group().casefold()
            for index in range(word_cursor, len(source_words)):
                if source_words[index].group().casefold() == target:
                    matched.append(index)
                    word_cursor = index + 1
                    break

        if not matched:
            result.append(chunk)
            continue

        last = matched[-1]
        start = source_char_cursor
        end = source_words[last].end()
        while end < len(source_text) and source_text[end] in ".,!?;:…)]}\"'’”":
            end += 1
        source = source_text[start:end].strip()
        result.append(source or chunk)
        source_cursor = last + 1
        source_char_cursor = end

    return result


def build_aligned_srt(
    text_chunks: list[str],
    audio_chunks: list[np.ndarray],
    sample_rate: int,
    *,
    silence_ps: list[float] | None = None,
    silence_p: float = 0.0,
    gap_samples: list[int] | None = None,
    display_chunks: list[str] | None = None,
    aligner=None,
    alignment_warnings: list[str] | None = None,
) -> str:
    """Use chunk timing and optional acoustic alignment for sentence boundaries."""
    if not text_chunks or len(text_chunks) != len(audio_chunks):
        raise ValueError("Số đoạn văn bản phải khớp với số đoạn audio.")
    if sample_rate <= 0:
        raise ValueError("Sample rate phải lớn hơn 0.")
    if silence_ps is not None and gap_samples is not None:
        raise ValueError("Chỉ truyền silence_ps hoặc gap_samples, không truyền đồng thời.")
    if gap_samples is not None and len(gap_samples) != len(text_chunks) - 1:
        raise ValueError("Số khoảng nghỉ phải bằng số ranh giới giữa các đoạn audio.")
    if silence_ps is not None and len(silence_ps) != len(text_chunks) - 1:
        raise ValueError("Số khoảng nghỉ phải bằng số ranh giới giữa các đoạn audio.")
    if display_chunks is not None and len(display_chunks) != len(text_chunks):
        raise ValueError("Số đoạn hiển thị phải bằng số đoạn văn bản.")

    gaps = []
    for index in range(len(text_chunks) - 1):
        if gap_samples is not None:
            gap = gap_samples[index]
        elif silence_ps is not None:
            gap = pause_pad_samples(
                np.asarray(audio_chunks[index], dtype=np.float32),
                np.asarray(audio_chunks[index + 1], dtype=np.float32),
                sample_rate,
                silence_ps[index],
            )
        else:
            gap = int(sample_rate * silence_p)
        if gap < 0:
            raise ValueError("Khoảng nghỉ audio không thể âm.")
        gaps.append(gap)

    entries = []
    alignment_failures = 0
    alignment_error = None
    cursor = 0
    for index, (text, audio) in enumerate(zip(text_chunks, audio_chunks)):
        samples = np.asarray(audio).size
        if samples <= 0:
            raise ValueError("Không thể tạo phụ đề cho đoạn audio rỗng.")

        captions = _captions_for_chunk(
            display_chunks[index] if display_chunks is not None else text
        )
        chunk_start = cursor

        aligned_spans = None
        if aligner is not None:
            try:
                aligned_spans = aligner.align_captions(
                    np.asarray(audio), sample_rate, captions
                )
                if len(aligned_spans) != len(captions):
                    raise ValueError("Số mốc căn chỉnh không khớp số câu.")
            except Exception as exc:
                alignment_error = str(exc)
                aligned_spans = None

        if aligned_spans is None:
            alignment_failures += 1
            weights = [max(1, len(caption)) for caption in captions]
            total_weight = sum(weights)
            elapsed_weight = 0
            for caption, weight in zip(captions, weights):
                elapsed_weight += weight
                start = chunk_start + round(samples * (elapsed_weight - weight) / total_weight)
                end = chunk_start + round(samples * elapsed_weight / total_weight)
                entries.append((caption, start, max(start + 1, end)))
        else:
            for caption, (start, end) in zip(captions, aligned_spans):
                local_start = min(samples - 1, max(0, int(start)))
                local_end = min(samples, max(local_start + 1, int(end)))
                entries.append(
                    (caption, chunk_start + local_start, chunk_start + local_end)
                )

        cursor += samples
        if index < len(gaps):
            cursor += gaps[index]

    if alignment_failures:
        if alignment_error:
            _report_alignment_warning(
                f"Căn chỉnh CTC thất bại ở {alignment_failures}/{len(text_chunks)} đoạn "
                f"({alignment_error}); các đoạn này dùng ước lượng theo ký tự.",
                alignment_warnings,
            )
        else:
            _report_alignment_warning(
                f"Căn chỉnh CTC thất bại ở {alignment_failures}/{len(text_chunks)} đoạn; "
                "các đoạn này dùng ước lượng theo ký tự.",
                alignment_warnings,
            )

    blocks = [
        f"{index}\n"
        f"{_format_timestamp(round(start * 1_000 / sample_rate))} --> "
        f"{_format_timestamp(round(end * 1_000 / sample_rate))}\n"
        f"{caption}"
        for index, (caption, start, end) in enumerate(entries, start=1)
    ]
    return "\n\n".join(blocks) + "\n"


def write_aligned_srt(
    audio_path: str,
    text_chunks: list[str],
    audio_chunks: list[np.ndarray],
    sample_rate: int,
    *,
    source_text: str | None = None,
    use_ctc_alignment: bool = False,
    alignment_warnings: list[str] | None = None,
    **join_options,
) -> str:
    """Write an SRT beside the audio file and return its path."""
    subtitle_path = os.path.splitext(audio_path)[0] + ".srt"
    display_chunks = None
    if source_text is not None:
        display_chunks = _source_chunks(source_text, text_chunks)

    aligner = None
    if use_ctc_alignment:
        try:
            from apps.ctc_alignment import VietnameseCTCAligner

            aligner = VietnameseCTCAligner.load()
        except Exception as exc:
            _report_alignment_warning(
                f"Không tải được mô hình căn chỉnh phụ đề ({exc}); "
                "phụ đề dùng ước lượng theo ký tự.",
                alignment_warnings,
            )

    subtitle = build_aligned_srt(
        text_chunks,
        audio_chunks,
        sample_rate,
        display_chunks=display_chunks,
        aligner=aligner,
        alignment_warnings=alignment_warnings,
        **join_options,
    )
    with open(subtitle_path, "w", encoding="utf-8-sig", newline="\n") as file:
        file.write(subtitle)
    return subtitle_path
