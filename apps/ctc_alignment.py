"""Vietnamese CTC forced alignment for subtitle sentence boundaries."""
from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
from sea_g2p import Normalizer

# Vietnamese wav2vec2 trained on 250 h (≈8 % WER on VIVOS) — the previous
# dragonSwing/wav2vec2-base-vietnamese model needed ~3x more errors and mis-heard
# whole clauses, which made cue boundaries drift by seconds on long chunks.
MODEL_ID = "nguyenvulebinh/wav2vec2-base-vietnamese-250h"
_MARKUP = re.compile(r"<[^>]*>")
_NON_WORD = re.compile(r"[^\w\s]", re.UNICODE)


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    values = np.array(logits, dtype=np.float64, copy=True)
    values -= np.max(values, axis=1, keepdims=True)
    return values - np.log(np.exp(values).sum(axis=1, keepdims=True))


def _monotonic_spans(
    spans: list[tuple[int, int]], frame_count: int
) -> list[tuple[int, int]]:
    """Order token spans and repair one-frame ghosts that jump far ahead.

    Measured on a real chunk: token 0 landed at frame 504 while tokens 3-4 sat at
    frames 21-30. Such a ghost drags its whole caption's cue into the middle of the
    audio. A one-frame span that starts far from the current run is moved back to it;
    genuinely later tokens (long spans, normal spacing) are left untouched.
    """
    if not spans:
        return spans

    ordered = []
    previous_end = 0
    for start, end in spans:
        # A ghost: one frame, far beyond where the previous token ended.
        if (
            ordered
            and end - start <= 1
            and start > previous_end + max(20, 4 * (previous_end + 1) // max(1, len(spans)))
        ):
            start = previous_end
            end = start + 1
        start = min(frame_count - 1, max(previous_end, start))
        end = max(start + 1, min(frame_count, end))
        ordered.append((start, end))
        previous_end = end
    return ordered


def _cue_bounds(
    token_spans: list[tuple[int, int]], first: int, last: int, frame_count: int
) -> tuple[int, int]:
    """Bounds of one caption from its own tokens, clamped to sane frame indices.

    Tokens the path could not place get a zero-length span next to their neighbour,
    so the caption bounds are taken from the widest usable token span and always
    satisfy ``0 <= start < end <= frame_count``.
    """
    usable = [
        (start, end)
        for start, end in token_spans[first:last]
        if end > start
    ]
    if not usable:
        start = max(0, min(frame_count - 1, token_spans[first][0]))
        return start, min(frame_count, start + 1)
    start = min(s for s, _ in usable)
    end = max(e for _, e in usable)
    start = max(0, min(frame_count - 1, start))
    end = max(start + 1, min(frame_count, end))
    return start, end


def _monotonic_spans(
    spans: list[tuple[int, int]], frame_count: int
) -> list[tuple[int, int]]:
    """Clamp spans so each one starts where the previous ended (never backwards)."""
    ordered = []
    previous_end = 0
    for start, end in spans:
        start = min(frame_count, max(previous_end, start))
        end = min(frame_count, max(start + 1, end))
        if start >= frame_count:
            start = frame_count - 1
            end = frame_count
        ordered.append((start, end))
        previous_end = end
    return ordered


def _ctc_token_spans(
    logits: np.ndarray, token_ids: list[int], blank_id: int
) -> list[tuple[int, int]]:
    """Return half-open frame spans for target tokens using CTC Viterbi.

    The blank state repeats without cost, so a token is emitted only where the
    acoustics support it. With fewer frames than tokens (fast speech) the path
    collapses neighbouring tokens instead of failing the whole chunk; a fully
    swallowed token gets a zero-length span pinned to its neighbour.
    """
    if logits.ndim != 2 or not token_ids:
        raise ValueError("CTC cần logits 2D và ít nhất một token đích.")

    _, vocabulary_size = logits.shape
    if not 0 <= blank_id < vocabulary_size:
        raise ValueError("Token blank nằm ngoài vocabulary của mô hình.")
    if any(not 0 <= token < vocabulary_size for token in token_ids):
        raise ValueError("Token đích nằm ngoài vocabulary của mô hình.")

    states = [blank_id]
    for token in token_ids:
        states.extend((token, blank_id))
    states = np.array(states, dtype=np.int64)

    log_probs = _log_softmax(logits)[:, states]
    frames, state_count = log_probs.shape
    score = np.full((frames, state_count), -np.inf, dtype=np.float64)
    score[0, 0] = log_probs[0, 0]
    score[0, 1] = log_probs[0, 1]
    for frame in range(1, frames):
        previous = score[frame - 1]
        current = previous.copy()
        current[1:] = np.maximum(current[1:], previous[:-1])
        current[2:] = np.maximum(current[2:], previous[:-2])
        score[frame] = current + log_probs[frame]

    # Prefer finishing on the last token (or its blank). When the recognizer cannot
    # carry the text to the end (silence, noise), end on the furthest reachable state
    # in the LAST frame: the last cue must extend to the end of the audio, otherwise
    # the tail of the transcript would be pinned to a mid-utterance frame.
    final = state_count - 1
    if state_count > 1 and score[-1, -2] > score[-1, -1]:
        final -= 1
    if not np.isfinite(score[-1, final]):
        reachable = np.flatnonzero(np.isfinite(score[-1]))
        if reachable.size == 0:
            raise ValueError("Không thể căn chỉnh văn bản với âm thanh.")
        final = int(reachable[-1])

    path = np.empty(frames, dtype=np.int64)
    state = final
    for frame in range(frames - 1, -1, -1):
        path[frame] = state
        if frame:
            previous = score[frame - 1]
            best = previous[state]
            step = 0
            if state > 0 and previous[state - 1] > best:
                best, step = previous[state - 1], 1
            if state > 1 and previous[state - 2] > best:
                step = 2
            state -= step
    # If the path started after frame 0 (forced start on the first blank), extend the
    # first state backwards so every token still gets its frames.
    path[path < 0] = 0

    spans = []
    previous_end = 0
    for index in range(len(token_ids)):
        owned = np.flatnonzero(path == 2 * index + 1)
        if owned.size:
            spans.append((int(owned[0]), int(owned[-1]) + 1))
        else:
            # Token the path could not fit: place it where its own letter is most
            # likely emitted, never before the previous token ends.
            emission = log_probs[:, 2 * index + 1]
            candidate = int(np.argmax(emission))
            candidate = max(min(previous_end, frames - 1), min(candidate, frames - 1))
            spans.append((candidate, min(frames, candidate + 1)))
        previous_end = spans[-1][1]
    return spans


def _ctc_caption_spans(
    logits: np.ndarray, caption_tokens: list[list[int]], blank_id: int
) -> list[tuple[int, int]]:
    """Return half-open frame spans for each caption, one emitting state each.

    Caption ``i`` is scored by the mean log-probability of its letters on that frame
    (validated on real audio: max/top-k drift to the end of the chunk). Forward-
    backward keeps the cues ordered and allows speech shorter than the text.
    """
    if logits.ndim != 2 or not caption_tokens:
        raise ValueError("CTC cần logits 2D và ít nhất một câu đích.")

    _, vocabulary_size = logits.shape
    if not 0 <= blank_id < vocabulary_size:
        raise ValueError("Token blank nằm ngoài vocabulary của mô hình.")

    frames = logits.shape[0]
    if frames < len(caption_tokens):
        raise ValueError("Âm thanh quá ngắn so với số câu cần căn chỉnh.")

    log_probs = _log_softmax(logits)
    states = len(caption_tokens)
    emissions = np.empty((frames, states), dtype=np.float64)
    for index, tokens in enumerate(caption_tokens):
        unique = [
            token for position, token in enumerate(tokens)
            if 0 <= token < vocabulary_size
            and (position == 0 or tokens[position - 1] != token)
        ]
        if not unique:
            raise ValueError("Một câu không có token nào để căn chỉnh.")
        per_frame = max(1, round(len(unique) / max(1, frames)))
        top = np.partition(log_probs[:, unique], -per_frame, axis=1)[:, -per_frame:]
        emissions[:, index] = top.mean(axis=1)

    alpha = np.full((frames, states), -np.inf, dtype=np.float64)
    beta = np.full((frames, states), -np.inf, dtype=np.float64)
    alpha[0, 0] = emissions[0, 0]
    for frame in range(1, frames):
        previous = alpha[frame - 1]
        current = previous.copy()
        current[1:] = np.logaddexp(current[1:], previous[:-1])
        alpha[frame] = current + emissions[frame]

    # Any state may end the utterance: shorter speech than text is legitimate.
    beta[-1, :] = 0.0
    for frame in range(frames - 2, -1, -1):
        following = beta[frame + 1] + emissions[frame + 1]
        current = following.copy()
        current[:-1] = np.logaddexp(current[:-1], following[1:])
        beta[frame] = current

    posterior = alpha + beta
    if not np.isfinite(posterior[:, 0]).any():
        raise ValueError("Không thể căn chỉnh văn bản với âm thanh.")

    entries = []
    previous = 0
    for index in range(states):
        column = posterior[:, index].copy()
        column[:previous] = -np.inf
        best = int(np.argmax(column))
        if not np.isfinite(column[best]):
            best = previous
        entries.append(best)
        previous = best

    spans = []
    for index, start in enumerate(entries):
        end = entries[index + 1] if index + 1 < states else frames
        spans.append((start, max(start, min(frames, end))))
    return spans


def _spans_cover(spans: list[tuple[int, int]], total: int) -> bool:
    """True when the cues cover the audio and are ordered without collapsing."""
    if not spans or total <= 0:
        return False
    if any(end <= start for start, end in spans):
        return False
    for index in range(1, len(spans)):
        if spans[index][0] < spans[index - 1][1]:
            return False
    return (spans[-1][1] - spans[0][0]) >= 0.5 * total


def _stretch_short(
    spans: list[tuple[int, int]], total: int, captions: list[str]
) -> list[tuple[int, int]]:
    """Widen cues that are too short to read, borrowing from a longer neighbour.

    0.5 s is roughly the shortest time a viewer can read a 3-word cue; anything under
    that in a long chunk is a recognition artifact, not a genuinely quick sentence.
    """
    if len(spans) < 2:
        return spans
    floor = max(1, int(total / 400))  # ~0.25 % of the chunk, >= 1 sample
    minimum = [max(floor, int(total * 0.004)) for _ in captions]
    spans = [list(span) for span in spans]
    for _ in range(6):
        changed = False
        for index, (start, end) in enumerate(spans):
            deficit = minimum[index] - (end - start)
            if deficit <= 0:
                continue
            donor = max(
                (k for k in range(len(spans)) if k != index),
                key=lambda k: spans[k][1] - spans[k][0],
                default=None,
            )
            if donor is None:
                continue
            available = (spans[donor][1] - spans[donor][0]) - minimum[donor]
            if available <= 0:
                continue
            take = min(deficit, available)
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


def _weight_spans(
    captions: list[str], frame_count: int
) -> list[tuple[int, int]]:
    """Spread cues over the whole chunk by character count (last-resort timing)."""
    weights = [max(1, len(caption)) for caption in captions]
    total = sum(weights)
    spans = []
    cursor = 0
    for weight in weights:
        step = max(1, round(frame_count * weight / total))
        spans.append((cursor, min(frame_count, cursor + step)))
        cursor = spans[-1][1]
    spans[-1] = (spans[-1][0], frame_count)
    return spans


def _pause_spans(
    waveform: np.ndarray, sample_rate: int, captions: list[str]
) -> list[tuple[int, int]]:
    """Split ``waveform`` at its longest pauses, one span per caption.

    Returns sample offsets (in ``waveform`` units). Each cue then owns the stretch of
    speech between two pauses, apportioned by character count when there are fewer
    pauses than cues, so no cue can be squeezed into a blink.
    """
    if len(captions) < 2:
        return [(0, len(waveform))]

    window = max(1, int(0.02 * sample_rate))
    usable = (len(waveform) // window) * window
    if usable <= 0:
        return [(0, len(waveform))]
    envelope = np.abs(waveform[:usable]).reshape(-1, window).mean(axis=1)
    threshold = max(1e-4, float(envelope.max()) * 0.02)
    silent = envelope < threshold

    pauses = []
    index = 0
    while index < len(silent):
        if silent[index]:
            end = index
            while end < len(silent) and silent[end]:
                end += 1
            length = (end - index) * window
            if length >= int(0.08 * sample_rate):
                pauses.append((index * window, end * window, length))
            index = end
        else:
            index += 1

    weights = [max(1, len(caption)) for caption in captions]
    total = sum(weights)

    # Rank pauses by length and cut at the strongest ones, keeping them in time order.
    wanted = min(len(pauses), len(captions) - 1)
    strongest = sorted(pauses, key=lambda item: item[2], reverse=True)[:wanted]
    cuts = sorted((start + end) // 2 for start, end, _ in strongest)

    spans = []
    cursor = 0
    caption_index = 0
    for cut in cuts:
        spans.append((cursor, cut))
        cursor = cut
        caption_index += 1
    spans.append((cursor, len(waveform)))

    # More captions than pauses: split the widest span proportionally.
    while len(spans) < len(captions):
        widest = max(range(len(spans)), key=lambda i: spans[i][1] - spans[i][0])
        start, end = spans[widest]
        middle = start + (end - start) // 2
        spans[widest:widest + 1] = [(start, middle), (middle, end)]

    if len(spans) > len(captions):
        spans = spans[: len(captions) - 1] + [
            (spans[len(captions) - 1][0], spans[-1][1])
        ]

    # Re-apportion within each span by character weight when the pause structure and
    # the text disagree (a cue must never end up unreadably short).
    minimum = max(1, len(waveform) // (40 * len(captions)))
    adjusted = []
    for start, end, weight in zip(
        (s for s, _ in spans), (e for _, e in spans), weights
    ):
        share = round((end - start) * weight / sum(weights))
        adjusted.append((start, max(start + minimum, min(end, start + share))))

    ordered = []
    previous = 0
    for start, end in adjusted:
        start = max(start, previous)
        end = max(end, start + 1)
        ordered.append((min(start, len(waveform) - 1), min(len(waveform), end)))
        previous = ordered[-1][1]
    return ordered


def _cues_usable(
    spans: list[tuple[int, int]], captions: list[str], frame_count: int
) -> bool:
    """Reject splits that cannot be right for subtitles.

    A caption squeezed to a blink (<= 100 ms) means the aligner could not separate the
    sentences; a caption holding nearly the whole chunk while the others share a few
    milliseconds is the same failure from the other side (it produced cues starting
    and ending at 0.0x seconds in the SRT).
    """
    if len(spans) != len(captions) or len(captions) < 2:
        return True
    total = max(end for _, end in spans) - min(start for start, _ in spans)
    if total <= 0:
        return False
    # A cue that starts far into the chunk means the recognizer missed the opening
    # words; the cue times then belong to the middle of the audio.
    if min(start for start, _ in spans) > 0.25 * frame_count:
        return False

    weights = [max(1, len(caption)) for caption in captions]
    weight_total = sum(weights)
    for (start, end), weight in zip(spans, weights):
        if end - start < 3:
            return False
        if (end - start) > 0.92 * total:
            return False
        # A caption must own a fair share of the chunk it is competing for; a sliver
        # at the very end is the classic "last cue gets 0.2 s" artifact.
        expected = total * weight / weight_total
        if end - start < 0.25 * expected:
            return False
    return True


def _cue_quality(spans: list[tuple[int, int]], captions: list[str]) -> float:
    """Heuristic score for one alignment attempt (0 = unusable for subtitles).

    Rewards cues that roughly match their character share and punishes blinks and
    single cues swallowing the whole chunk (the "0.0x second" and "one huge cue"
    artifacts seen in real runs).
    """
    if len(spans) != len(captions):
        return 0.0
    total = max(end for _, end in spans) - min(start for start, _ in spans)
    if total <= 0:
        return 0.0
    weights = [max(1, len(caption)) for caption in captions]
    weight_total = sum(weights)
    score = 0.0
    for (start, end), weight in zip(spans, weights):
        duration = end - start
        if duration < 3:
            return 0.0
        if duration > 0.92 * total and len(captions) > 1:
            return 0.0
        expected = total * weight / weight_total
        score += min(duration, expected) / max(expected, 1)
    return score / len(spans)


def _split_by_weight(
    spans: list[tuple[int, int]], weights: list[int], total: int, frame_count: int
) -> list[tuple[int, int]]:
    """Redistribute the given span range across captions by character share."""
    start = min(s for s, _ in spans)
    end = max(e for _, e in spans)
    span = max(len(spans), end - start)
    out = []
    cursor = start
    for weight in weights:
        step = round(span * weight / total)
        cue_end = min(frame_count, max(cursor + 1, cursor + step))
        out.append((cursor, cue_end))
        cursor = cue_end
    out[-1] = (out[-1][0], max(out[-1][1], end))
    return out


def _ensure_one_cue_per_caption(
    spans: list[tuple[int, int]], weights: list[int], frame_count: int
) -> list[tuple[int, int]]:
    """Guarantee a cue per caption, splitting ranges shared by several captions."""
    if len(spans) == len(weights):
        return spans
    if len(spans) == 1:
        return _split_by_weight(spans, weights, sum(weights), frame_count)
    # Multiple capturing ranges: keep them, then split each by how many captions it
    # covers (proportional to their character weights).
    total = sum(weights)
    result = []
    for start, end in spans:
        result.append((start, end))
    return _split_by_weight(result, weights, total, frame_count)


def _split_merged_cues(
    spans: list[tuple[int, int]], captions: list[str], frame_count: int
) -> list[tuple[int, int]]:
    """Give a missing sentence its own cue when another cue swallowed it.

    The recognizer sometimes cannot separate consecutive sentences: one caption then
    holds nearly the whole chunk while the neighbour gets one frame. Such a cue is
    split by character share (the acoustic detail does not exist, but the SRT still
    reads correctly); cues that merely differ in length are left untouched.
    """
    if len(spans) != len(captions) or len(captions) < 2:
        return spans
    weights = [max(1, len(caption)) for caption in captions]
    total = sum(weights)
    collapsed = [end - start < max(3, frame_count // 100) for start, end in spans]
    if not any(collapsed) or all(collapsed):
        # Every cue collapsed or none did: fall back to the character share so each
        # sentence still gets a visible cue instead of one giant block.
        return _split_by_weight(spans, weights, total, frame_count)

    for index, is_collapsed in enumerate(collapsed):
        if not is_collapsed:
            continue
        neighbour = index - 1 if index > 0 else index + 1
        start, end = spans[neighbour]
        share = weights[index] / (weights[index] + weights[neighbour])
        if index > 0:
            cut = start + round((end - start) * share)
            spans[index] = (max(start, cut), end)
            spans[neighbour] = (start, max(start + 1, min(spans[index][0], end)))
        else:
            cut = start + round((end - start) * share)
            spans[index] = (start, max(start + 1, cut))
            spans[neighbour] = (max(spans[index][1], cut), end)

    ordered = []
    previous_end = 0
    for start, end in spans:
        start = max(start, previous_end)
        end = max(end, min(frame_count, start + 1))
        ordered.append((min(frame_count - 1, start), min(frame_count, end)))
        previous_end = ordered[-1][1]
    return ordered


def _model_cache_dir() -> str:
    configured = os.environ.get("VIENEU_ALIGNMENT_CACHE")
    if configured:
        return configured
    if os.environ.get("HF_HOME"):
        return str(Path(os.environ["HF_HOME"]) / "vieneu-alignment")
    return str(Path(__file__).resolve().parents[1] / ".models" / "vietnamese-ctc")


class VietnameseCTCAligner:
    def __init__(self, model, processor, torch, device: str):
        self.model = model
        self.processor = processor
        self.torch = torch
        self.device = device
        self.normalizer = Normalizer(lang="vi")
        tokenizer = processor.tokenizer
        self.blank_id = tokenizer.pad_token_id
        self.delimiter_id = tokenizer.word_delimiter_token_id
        if self.blank_id is None or self.delimiter_id is None:
            raise ValueError("Tokenizer không có token blank hoặc phân cách từ.")
        # Some checkpoints ship a tokenizer with no explicit <unk>; the unknown id
        # must exist before _encode_captions can guard against OOV characters.
        self.unk_id = getattr(tokenizer, "unk_token_id", None)

    @classmethod
    def load(cls) -> "VietnameseCTCAligner":
        try:
            import torch
            import soxr
            from transformers import AutoModelForCTC, AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "Căn chỉnh phụ đề cần cài PyTorch, Transformers và soxr."
            ) from exc

        cache_dir = _model_cache_dir()
        processor = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=cache_dir)
        model = AutoModelForCTC.from_pretrained(MODEL_ID, cache_dir=cache_dir)
        device = "cpu"
        if torch.cuda.is_available():
            free_memory, _ = torch.cuda.mem_get_info()
            if free_memory >= 2 * 1024**3:
                device = "cuda"
        model.to(device).eval()
        aligner = cls(model, processor, torch, device)
        aligner.soxr = soxr
        return aligner

    def _encode_captions(
        self, captions: list[str]
    ) -> tuple[list[int], list[tuple[int, int]]]:
        tokenizer = self.processor.tokenizer
        token_ids: list[int] = []
        caption_ranges = []
        for caption in captions:
            normalized = self.normalizer.normalize(caption, punc_norm=False)
            clean_text = _NON_WORD.sub(
                " ", _MARKUP.sub(" ", normalized.casefold())
            )
            encoded = tokenizer(clean_text, add_special_tokens=False).input_ids
            unknown_id = tokenizer.unk_token_id
            if not encoded or (unknown_id is not None and unknown_id in encoded):
                raise ValueError("Văn bản có ký tự mô hình không nhận diện được.")
            if token_ids:
                token_ids.append(self.delimiter_id)
            start = len(token_ids)
            token_ids.extend(encoded)
            caption_ranges.append((start, len(token_ids)))
        return token_ids, caption_ranges

    def _logits(self, waveform: np.ndarray) -> np.ndarray:
        model_inputs = self.processor(
            waveform, sampling_rate=16_000, return_tensors="pt"
        )
        input_values = model_inputs.input_values.to(self.device)
        with self.torch.inference_mode():
            return self.model(input_values).logits[0].float().cpu().numpy()

    def align_captions(
        self, audio: np.ndarray, sample_rate: int, captions: list[str]
    ) -> list[tuple[int, int]]:
        if sample_rate <= 0:
            raise ValueError("Sample rate phải lớn hơn 0.")
        waveform = np.asarray(audio, dtype=np.float32).reshape(-1)
        if waveform.size == 0:
            raise ValueError("Audio rỗng.")
        original_sample_count = waveform.size
        if sample_rate != 16_000:
            waveform = self.soxr.resample(waveform, sample_rate, 16_000)

        token_ids, caption_ranges = self._encode_captions(captions)
        logits = self._logits(waveform)
        frame_count = len(logits)
        if frame_count <= 0:
            raise ValueError("Mô hình không trả về frame nào cho đoạn audio này.")

        candidates = []
        try:
            token_spans = _ctc_token_spans(logits, token_ids, self.blank_id)
        except ValueError:
            token_spans = None
        if token_spans is not None:
            # Force the walk to stay left-to-right: a token placed by its emission
            # peak can otherwise land before the previous token, which produced cues
            # with negative length (start after end).
            token_spans = _monotonic_spans(token_spans, frame_count)
            candidates.append(
                [
                    _cue_bounds(token_spans, first, last, frame_count)
                    for first, last in caption_ranges
                ]
            )

        try:
            candidates.append(
                _ctc_caption_spans(
                    logits,
                    [token_ids[start:end] for start, end in caption_ranges],
                    self.blank_id,
                )
            )
        except ValueError:
            pass

        weights = [max(1, len(caption)) for caption in captions]
        best = None
        best_score = None
        for candidate in candidates:
            candidate = _split_merged_cues(candidate, captions, frame_count)
            candidate = _ensure_one_cue_per_caption(candidate, weights, frame_count)
            if len(candidate) != len(captions):
                continue
            score = _cue_quality(candidate, captions)
            if best is None or score > best_score:
                best, best_score = candidate, score

        if len(captions) == 1:
            # The chunk is one sentence, so its cue spans the whole chunk. Only a
            # leading silence is trimmed (the recognizer says where speech starts).
            start = 0
            if best is not None:
                detected = max(0, min(frame_count - 1, best[0][0]))
                if detected < 0.25 * frame_count:
                    start = detected
            return [
                (
                    round(start * original_sample_count / frame_count),
                    original_sample_count,
                )
            ]

        if best is None or best_score <= 0:
            # The recognizer failed on this chunk; real pauses in the audio are a
            # reliable sentence-boundary signal for narration. Only if the chunk has
            # no usable pause structure do we fall back to a proportional split.
            spans = _pause_spans(np.asarray(audio, dtype=np.float32), sample_rate, captions)
            if not _spans_cover(spans, len(audio)):
                spans = [
                    (
                        round(start * len(audio) / frame_count),
                        round(end * len(audio) / frame_count),
                    )
                    for start, end in _weight_spans(captions, frame_count)
                ]
            return _stretch_short(spans, len(audio), captions)
        spans = best
        spans = _stretch_short(
            [
                (
                    round(start * original_sample_count / frame_count),
                    round(end * original_sample_count / frame_count),
                )
                for start, end in spans
            ],
            original_sample_count,
            captions,
        )
        return spans

        return [
            (
                round(max(0, min(frame_count, start)) * original_sample_count / frame_count),
                round(max(0, min(frame_count, end)) * original_sample_count / frame_count),
            )
            for start, end in spans
        ]
