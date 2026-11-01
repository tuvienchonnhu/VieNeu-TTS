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
# Sentence ends inside a normalized chunk ("cau mot. cau hai.").
_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")
_LOGGER = logging.getLogger(__name__)
# A caption shorter than this fraction of its character-proportional share is
# treated as a misalignment (one caption eating the whole chunk).
_MIN_SPAN_RATIO = 0.25
# The CTC aligner (~400 MB) is loaded once and reused: every extra subtitle run used
# to reload it, which cost seconds per generate click.
_ALIGNER = None
_ALIGNER_ERROR = None

# --- subtitle readability -------------------------------------------------------
# One sentence per cue, never more than this many words.
MAX_CUE_WORDS = 16
_LINE_BREAK_AFTER = (".", "?", "!", ",", ";", ":", "…")
# A cue may not start with a fragment that belongs to the previous clause.
_LEADING_OK = {
    "và", "hoặc", "nhưng", "mà", "nên", "vì", "thì", "rồi", "hay",
    "cho", "để", "khi", "nếu", "tuy", "nhờ", "do", "với", "của",
}


def _load_aligner():
    """Return the shared VietnameseCTCAligner, loading it on first use."""
    global _ALIGNER, _ALIGNER_ERROR
    if _ALIGNER is None:
        if _ALIGNER_ERROR is not None:
            raise RuntimeError(_ALIGNER_ERROR)
        try:
            from apps.ctc_alignment import VietnameseCTCAligner

            _ALIGNER = VietnameseCTCAligner.load()
        except Exception as exc:
            # Keep the failure: retrying a missing dependency on every click only
            # repeats the same download/import error.
            _ALIGNER_ERROR = str(exc)
            raise
    return _ALIGNER


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


def _valid_spans(
    spans, samples: int, captions: list[str]
) -> list[tuple[int, int]] | None:
    """Normalise the aligner response: integers, right count, inside the chunk.

    Ordering and minimum length are not checked here - the final pass repairs them -
    so a slightly odd acoustic result no longer discards a whole chunk's timing.
    """
    try:
        spans = [(int(start), int(end)) for start, end in spans]
    except (TypeError, ValueError):
        return None
    if len(spans) != len(captions):
        return None
    if any(start < 0 or end > samples for start, end in spans):
        return None
    return spans


def _split_long_caption(caption: str, max_words: int = MAX_CUE_WORDS) -> list[str]:
    """Split a sentence into cues of at most ``max_words`` words.

    Priority: punctuation, then a conjunction/preposition. Only when neither exists
    inside the window is the clause cut near the middle. A piece shorter than four
    words is merged back into its neighbour, so cues read as phrases.
    """
    words = caption.split()
    if len(words) <= max_words:
        return [caption]

    pieces: list[str] = []
    remaining = words
    while len(remaining) > max_words:
        window = remaining[:max_words]
        cut = None
        # Punctuation wins, but only if the left part stays substantial.
        for index in range(len(window), 1, -1):
            if window[index - 1].endswith(_LINE_BREAK_AFTER):
                cut = index
                break
        if cut is None:
            for index in range(len(window) - 1, 1, -1):
                if window[index].casefold().strip(",;:") in _LEADING_OK:
                    cut = index
                    break
        if cut is None:
            cut = max_words
        # Never emit a stub: the left piece must carry at least half the cue.
        if cut < max(4, max_words // 2):
            cut = max_words
        if len(remaining) - cut < 4:
            cut = max(1, len(remaining) - 4)
        pieces.append(" ".join(remaining[:cut]))
        remaining = remaining[cut:]
    if remaining:
        pieces.append(" ".join(remaining))

    # Merge a 1-3 word piece into its neighbour: a fragment on screen is worse than
    # one slightly longer cue.
    merged: list[str] = []
    for piece in pieces:
        too_short = len(piece.split()) < 4
        previous = merged[-1] if merged else None
        if previous is not None and (too_short or len(previous.split()) < 4):
            merged[-1] = previous + " " + piece
        else:
            merged.append(piece)

    # A merge may exceed the cap; split that one again without the short-merge rule.
    final: list[str] = []
    for piece in merged:
        if len(piece.split()) <= max_words:
            final.append(piece)
        else:
            words = piece.split()
            while len(words) > max_words:
                final.append(" ".join(words[:max_words]))
                words = words[max_words:]
            if words:
                final.append(" ".join(words))
    return final


def _captions_for_chunk(text: str) -> list[str]:
    """One sentence per caption, split further so no cue exceeds MAX_CUE_WORDS.

    Normalized v3 chunks arrive as ``"cau mot. cau hai."`` - several sentences in one
    lower-cased string - so a period followed by a space is a sentence end too.
    """
    pieces = []
    for paragraph in split_text_into_paragraphs(text):
        sentences = split_into_sentences(paragraph)
        if len(sentences) <= 1:
            # No punctuation-driven split (or one big sentence): use the periods.
            sentences = [
                part.strip()
                for part in _SENTENCE_END.split(paragraph)
                if part.strip()
            ] or sentences
        pieces.extend(sentences)

    captions = []
    for sentence in pieces:
        # Keep the terminating punctuation with its sentence.
        sentence = sentence.strip()
        if sentence and sentence[-1] not in ".!?…":
            sentence += "."
        captions.extend(_split_long_caption(sentence))
    return [caption.strip() for caption in captions if caption.strip()] or [text.strip()]


def _source_chunks(source_text: str, normalized_chunks: list[str]) -> list[str]:
    """Recover the original spelling of each chunk, in order and never overlapping.

    The normalized text decides *how many* words belong to a chunk; the source text
    supplies *which characters* they are. This keeps original case, punctuation and
    numerals, and cannot duplicate text (the old version scanned forward and, when a
    rewritten word was not found, swallowed the rest of the document into one chunk).
    """
    source_words = list(_WORD.finditer(source_text))
    if not source_words:
        return list(normalized_chunks)

    result = []
    cursor = 0          # word index in source_text
    char_cursor = 0     # character index in source_text

    for position, chunk in enumerate(normalized_chunks):
        remaining_words = len(source_words) - cursor
        remaining_chunks = len(normalized_chunks) - position
        if remaining_words <= 0:
            result.append(chunk)
            continue

        if position == len(normalized_chunks) - 1:
            take = remaining_words
        else:
            target = len(_WORD.findall(chunk))
            take = max(1, min(target, remaining_words - (remaining_chunks - 1)))
        take = min(take, remaining_words)

        last = cursor + take - 1
        start = char_cursor
        end = source_words[last].end()
        while end < len(source_text) and source_text[end] in ".,!?;:\u2026)]}\"'\u2019\u201d":
            end += 1

        text = source_text[start:end].strip()
        result.append(text or chunk)

        cursor = last + 1
        char_cursor = end
        while char_cursor < len(source_text) and source_text[char_cursor].isspace():
            char_cursor += 1

    return result


def _repair_cue_clusters(
    entries: list[tuple[str, int, int]], sample_rate: int
) -> list[tuple[str, int, int]]:
    """Spread a run of cues that were all pushed into a few seconds.

    The recognizer sometimes pins several minutes of speech to a single moment, so
    cues like "00:00:48,640 --> 00:00:48,901" appear next to a 16-second cue. Each
    such burst is re-timed inside its own window, never leaving its window, and keeps
    the reading speed of the surrounding text.
    """
    if len(entries) < 2:
        return entries
    # A cue is "too fast" when it holds more words than can be read in its time.
    readable_per_frame = sample_rate * 0.32  # samples per word of normal reading
    out = []
    index = 0
    while index < len(entries):
        text, start, end = entries[index]
        words = max(1, len(text.split()))
        if (end - start) >= min(words * readable_per_frame, 0.35 * sample_rate) or index == 0:
            out.append((text, start, end))
            index += 1
            continue

        # Collect the consecutive cues that are all unreadably fast.
        burst = []
        cursor = index
        while cursor < len(entries):
            caption, cue_start, cue_end = entries[cursor]
            count = max(1, len(caption.split()))
            if (cue_end - cue_start) >= min(
                count * readable_per_frame, 0.35 * sample_rate
            ):
                break
            burst.append(entries[cursor])
            cursor += 1

        window_start = out[-1][1] if out else burst[0][1]
        next_start = entries[cursor][1] if cursor < len(entries) else burst[-1][2]
        window = max(next_start - window_start, len(burst) * int(0.35 * sample_rate))
        weights = [max(1, len(caption.split())) for caption, _, _ in burst]
        total = sum(weights)
        position = window_start
        for (caption, _, _), weight in zip(burst, weights):
            step = max(1, round(window * weight / total))
            out.append((caption, position, position + step))
            position += step
        index = cursor

    return out


def _widen_short_cues(
    spans: list[tuple[int, int]], captions: list[str], samples: int
) -> list[tuple[int, int]]:
    """Give any cue under ~0.4 s some room by borrowing from a longer neighbour.

    The recogniser can leave a 0.2 s cue when two sentences are merged; such a cue is
    unreadable in the SRT. Only the neighbouring cues inside the same chunk are used,
    so later chunks keep their timing.
    """
    if len(spans) < 2 or samples <= 0:
        return spans
    floor = max(1, int(0.4 * samples / max(1, len(captions))))
    spans = [list(span) for span in spans]
    for _ in range(4):
        changed = False
        for index, (start, end) in enumerate(spans):
            deficit = floor - (end - start)
            if deficit <= 0:
                continue
            donor = max(
                (k for k in range(len(spans)) if k != index and spans[k][1] - spans[k][0] > floor + deficit),
                key=lambda k: spans[k][1] - spans[k][0],
                default=None,
            )
            if donor is None:
                continue
            take = min(deficit, (spans[donor][1] - spans[donor][0]) - floor)
            if take <= 0:
                continue
            if donor < index:
                spans[index][0] -= take
                spans[donor][1] -= take
            else:
                spans[index][1] += take
                spans[donor][0] += take
            changed = True
        if not changed:
            break

    ordered = []
    previous = 0
    for start, end in spans:
        start = max(previous, max(0, start))
        end = max(end, start + 1)
        ordered.append((start, end))
        previous = end
    return ordered


def _finish_entries(
    entries: list[tuple[str, int, int]], sample_rate: int
) -> list[tuple[str, int, int]]:
    """Last guarantee for the SRT: readable cue length and strictly increasing times."""
    if not entries:
        return entries
    total = entries[-1][2] - entries[0][1]
    floor = max(1, int(0.35 * sample_rate))
    fixed = [(text, start, end) for text, start, end in entries]
    for index, (text, start, end) in enumerate(fixed):
        deficit = floor - (end - start)
        if deficit <= 0:
            continue
        donor = max(
            (
                k for k in range(len(fixed))
                if k != index and (fixed[k][2] - fixed[k][1]) > floor + deficit
            ),
            key=lambda k: fixed[k][2] - fixed[k][1],
            default=None,
        )
        if donor is None:
            continue
        take = min(deficit, (fixed[donor][2] - fixed[donor][1]) - floor)
        if take <= 0:
            continue
        if donor < index:
            fixed[index] = (text, start - take, end)
            fixed[donor] = (fixed[donor][0], fixed[donor][1], fixed[donor][2] - take)
        else:
            fixed[index] = (text, start, end + take)
            fixed[donor] = (fixed[donor][0], fixed[donor][1] + take, fixed[donor][2])

    ordered = []
    cursor = 0
    for text, start, end in fixed:
        start = max(start, cursor)
        end = max(end, start + 1)
        ordered.append((text, start, end))
        cursor = end
    return ordered


def _wrap_caption(caption: str) -> str:
    """Break a cue into at most two readable lines, never splitting a word."""
    words = caption.split()
    if len(words) <= 8:
        return caption
    half = (len(words) + 1) // 2
    for index in range(half, 0, -1):
        if words[index - 1].endswith(_LINE_BREAK_AFTER):
            half = index
            break
    return " ".join(words[:half]) + "\n" + " ".join(words[half:])


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
        alignment_attempted = aligner is not None
        if alignment_attempted:
            try:
                aligned_spans = aligner.align_captions(
                    np.asarray(audio), sample_rate, captions
                )
                if len(aligned_spans) != len(captions):
                    raise ValueError("Số mốc căn chỉnh không khớp số câu.")
                # align_captions already guarantees ordered, covering cues; this
                # check only normalises types and rejects a broken response.
                aligned_spans = _valid_spans(aligned_spans, samples, captions)
                if aligned_spans is None:
                    raise ValueError("Mốc căn chỉnh không khớp số câu.")
            except Exception as exc:
                alignment_error = str(exc)
                aligned_spans = None

        if aligned_spans is None:
            alignment_failures += alignment_attempted
            weights = [max(1, len(caption)) for caption in captions]
            total_weight = sum(weights)
            elapsed_weight = 0
            chunk_spans = []
            for weight in weights:
                elapsed_weight += weight
                start = round(samples * (elapsed_weight - weight) / total_weight)
                end = round(samples * elapsed_weight / total_weight)
                chunk_spans.append((start, max(start + 1, end)))
        else:
            chunk_spans = []
            for start, end in aligned_spans:
                local_start = min(samples - 1, max(0, int(start)))
                local_end = min(samples, max(local_start + 1, int(end)))
                chunk_spans.append((local_start, local_end))

        for caption, (start, end) in zip(captions, chunk_spans):
            entries.append((caption, chunk_start + start, chunk_start + end))

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

    entries = _repair_cue_clusters(entries, sample_rate)
    # Last safety net: cues must never overlap or run backwards in an SRT.
    cursor = 0
    ordered_entries = []
    for caption, start, end in entries:
        start = max(start, cursor)
        end = max(end, start + 1)
        ordered_entries.append((caption, start, end))
        cursor = end
    entries = ordered_entries

    entries = _finish_entries(entries, sample_rate)

    blocks = [
        f"{index}\n"
        f"{_format_timestamp(round(start * 1_000 / sample_rate))} --> "
        f"{_format_timestamp(round(end * 1_000 / sample_rate))}\n"
        f"{_wrap_caption(caption)}"
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
            aligner = _load_aligner()
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
