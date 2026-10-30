"""Vietnamese CTC forced alignment for subtitle sentence boundaries."""
from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
from sea_g2p import Normalizer

MODEL_ID = "dragonSwing/wav2vec2-base-vietnamese"
_MARKUP = re.compile(r"<[^>]*>")
_NON_WORD = re.compile(r"[^\w\s]", re.UNICODE)


def _ctc_token_spans(
    logits: np.ndarray, token_ids: list[int], blank_id: int
) -> list[tuple[int, int]]:
    """Return half-open frame spans for target tokens using CTC Viterbi."""
    logits = np.array(logits, dtype=np.float64, copy=True)
    if logits.ndim != 2 or not token_ids:
        raise ValueError("CTC cần logits 2D và ít nhất một token đích.")

    frames, vocabulary_size = logits.shape
    if blank_id < 0 or blank_id >= vocabulary_size:
        raise ValueError("Token blank nằm ngoài vocabulary của mô hình.")
    if any(token < 0 or token >= vocabulary_size for token in token_ids):
        raise ValueError("Token đích nằm ngoài vocabulary của mô hình.")

    expanded = np.zeros(2 * len(token_ids) + 1, dtype=np.int64)
    expanded[1::2] = token_ids
    states = len(expanded)
    if frames < len(token_ids):
        raise ValueError("Âm thanh quá ngắn so với văn bản cần căn chỉnh.")

    logits -= np.max(logits, axis=1, keepdims=True)
    log_probs = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))

    previous = np.full(states, -np.inf, dtype=np.float64)
    previous[0] = log_probs[0, blank_id]
    previous[1] = log_probs[0, expanded[1]]
    backpointers = np.zeros((frames, states), dtype=np.uint8)
    state_indices = np.arange(states)
    skip_allowed = np.zeros(states, dtype=bool)
    skip_allowed[2:] = (
        (expanded[2:] != blank_id)
        & (expanded[2:] != expanded[:-2])
    )

    for frame in range(1, frames):
        candidates = np.full((3, states), -np.inf, dtype=np.float64)
        candidates[0] = previous
        candidates[1, 1:] = previous[:-1]
        candidates[2, 2:] = previous[:-2]
        candidates[2, ~skip_allowed] = -np.inf
        choices = np.argmax(candidates, axis=0)
        previous = candidates[choices, state_indices] + log_probs[
            frame, expanded
        ]
        backpointers[frame] = choices

    final_state = states - 1
    if states > 1 and previous[-2] > previous[-1]:
        final_state -= 1
    if not np.isfinite(previous[final_state]):
        raise ValueError("Không thể căn chỉnh văn bản với âm thanh.")

    state_path = np.empty(frames, dtype=np.int64)
    state = final_state
    for frame in range(frames - 1, -1, -1):
        state_path[frame] = state
        if frame:
            state -= int(backpointers[frame, state])

    spans = []
    for token_index in range(len(token_ids)):
        token_frames = np.flatnonzero(state_path == 2 * token_index + 1)
        if token_frames.size == 0:
            raise ValueError("Không tìm thấy thời điểm cho một token.")
        spans.append((int(token_frames[0]), int(token_frames[-1]) + 1))
    return spans


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

        model_inputs = self.processor(
            waveform, sampling_rate=16_000, return_tensors="pt"
        )
        input_values = model_inputs.input_values.to(self.device)
        with self.torch.inference_mode():
            logits = self.model(input_values).logits[0].float().cpu().numpy()

        token_spans = _ctc_token_spans(logits, token_ids, self.blank_id)
        frame_count = len(logits)
        return [
            (
                round(token_spans[first][0] * original_sample_count / frame_count),
                round(token_spans[last - 1][1] * original_sample_count / frame_count),
            )
            for first, last in caption_ranges
        ]
