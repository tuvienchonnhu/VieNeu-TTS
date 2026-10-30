"""Create subtitles aligned to the generated audio chunks."""
from __future__ import annotations

import os
import re

import numpy as np

from vieneu_utils.core_utils import (
    pack_sentences_into_chunks,
    pause_pad_samples,
    split_into_sentences,
    split_text_into_paragraphs,
)

_WORD = re.compile(r"[^\W_]+", re.UNICODE)


def _format_timestamp(milliseconds: int) -> str:
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, milliseconds = divmod(remainder, 1_000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"


def _captions_for_chunk(text: str) -> list[str]:
    captions = []
    for paragraph in split_text_into_paragraphs(text):
        sentences = split_into_sentences(paragraph)
        for sentence in sentences:
            captions.extend(pack_sentences_into_chunks([sentence], max_chars=84))
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
) -> str:
    """Use actual chunk durations and join pauses; divide each chunk among its sentences."""
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
    cursor = 0
    for index, (text, audio) in enumerate(zip(text_chunks, audio_chunks)):
        samples = np.asarray(audio).size
        if samples <= 0:
            raise ValueError("Không thể tạo phụ đề cho đoạn audio rỗng.")

        captions = _captions_for_chunk(text)
        weights = [max(1, len(caption)) for caption in captions]
        total_weight = sum(weights)
        chunk_start = cursor
        elapsed_weight = 0
        for caption, weight in zip(captions, weights):
            elapsed_weight += weight
            start = chunk_start + round(samples * (elapsed_weight - weight) / total_weight)
            end = chunk_start + round(samples * elapsed_weight / total_weight)
            entries.append((caption, start, max(start + 1, end)))

        cursor += samples
        if index < len(gaps):
            cursor += gaps[index]

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
    **join_options,
) -> str:
    """Write an SRT beside the audio file and return its path."""
    subtitle_path = os.path.splitext(audio_path)[0] + ".srt"
    if source_text is not None:
        text_chunks = _source_chunks(source_text, text_chunks)
    subtitle = build_aligned_srt(
        text_chunks, audio_chunks, sample_rate, **join_options
    )
    with open(subtitle_path, "w", encoding="utf-8-sig", newline="\n") as file:
        file.write(subtitle)
    return subtitle_path
