from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from apps.ctc_alignment import VietnameseCTCAligner, _ctc_caption_spans, _ctc_token_spans


def test_ctc_viterbi_recovers_known_token_frame_spans():
    logits = np.zeros((5, 3), dtype=np.float32)
    logits[0:2, 1] = 8
    logits[2, 0] = 8
    logits[3:5, 2] = 8

    assert _ctc_token_spans(logits, [1, 2], blank_id=0) == [(0, 2), (3, 5)]


def test_ctc_viterbi_requires_blank_between_repeated_tokens():
    logits = np.zeros((4, 2), dtype=np.float32)
    logits[0, 1] = 8
    logits[1, 0] = 8
    logits[2:4, 1] = 8

    assert _ctc_token_spans(logits, [1, 1], blank_id=0) == [(0, 1), (2, 4)]


def test_ctc_caption_spans_handle_fewer_frames_than_characters():
    # 3 frames only, but captions of several tokens each (fast speech): per-token
    # Viterbi cannot fit, caption-level Viterbi must still split the audio in order.
    logits = np.full((3, 40), -8.0, dtype=np.float32)
    logits[:, 0] = 0.0
    logits[0, 11:13] = 8.0
    logits[1, 11:13] = 4.0
    logits[1, 21:23] = 8.0
    logits[2, 31:33] = 8.0
    captions = [[11, 12], [21, 22], [31, 32]]

    spans = _ctc_caption_spans(logits, captions, blank_id=0)

    assert spans[0][0] == 0
    assert spans[1][0] >= spans[0][1]
    assert spans[2][1] == 3


def test_ctc_caption_spans_reject_more_captions_than_frames():
    logits = np.zeros((2, 40), dtype=np.float32)
    with pytest.raises(ValueError, match="quá ngắn"):
        _ctc_caption_spans(logits, [[11], [12], [13]], blank_id=0)


def test_ctc_viterbi_tolerates_fewer_frames_than_tokens():
    # Fast speech: 6 frames for 9 tokens. The old code raised "Am thanh qua ngan"
    # and threw the whole chunk back to a character estimate.
    tokens = list(range(11, 20))
    logits = np.full((6, 30), -8.0, dtype=np.float32)
    logits[:, 0] = 0.0
    # Each frame strongly supports one later token; blanks carry the rest.
    for frame, token in enumerate(tokens[:6]):
        logits[frame, token] = 8.0

    spans = _ctc_token_spans(logits, tokens, blank_id=0)

    assert len(spans) == len(tokens)
    assert all(0 <= start <= end <= 6 for start, end in spans)
    assert [start for start, _ in spans] == sorted(start for start, _ in spans)
    # The last token must still claim the end of the audio (cue boundary).
    assert spans[-1][1] == 6


def test_ctc_viterbi_keeps_alignment_when_a_token_is_missed():
    # Token 2 has no acoustic support: the neighbours keep their real frames instead
    # of the whole alignment being abandoned.
    logits = np.full((8, 5), -8.0, dtype=np.float32)
    logits[:, 0] = 0.0
    logits[1:4, 1] = 8.0
    logits[5:8, 3] = 8.0

    spans = _ctc_token_spans(logits, [1, 2, 3], blank_id=0)

    assert spans[0][0] == 1
    assert spans[-1][1] == 8
    assert spans[1][0] == spans[1][1] or spans[1][1] <= spans[2][1]


def test_aligner_returns_sample_offsets_at_original_sample_rate():
    class FakeTensor:
        def __init__(self, value):
            self.value = value

        def to(self, device):
            return self

        def float(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.value

        def __getitem__(self, index):
            return FakeTensor(self.value[index])

    class FakeTokenizer:
        pad_token_id = 0
        word_delimiter_token_id = 3
        unk_token_id = 4

        def __call__(self, text, add_special_tokens):
            return SimpleNamespace(input_ids=[1, 2])

    class FakeProcessor:
        tokenizer = FakeTokenizer()

        def __call__(self, waveform, sampling_rate, return_tensors):
            return SimpleNamespace(input_values=FakeTensor(np.zeros((1, len(waveform)))))

    logits = np.zeros((50, 4), dtype=np.float32)
    logits[0, 0] = 8
    logits[1:12, 1] = 8
    logits[12, 0] = 8
    logits[13:25, 2] = 8

    class FakeModel:
        def __call__(self, input_values):
            return SimpleNamespace(logits=FakeTensor(logits[np.newaxis, ...]))

    class FakeTorch:
        @staticmethod
        def inference_mode():
            return nullcontext()

    import soxr

    aligner = VietnameseCTCAligner(
        FakeModel(), FakeProcessor(), FakeTorch(), "cpu"
    )
    aligner.soxr = soxr
    spans = aligner.align_captions(
        np.ones(24_000, dtype=np.float32), 24_000, ["Một câu."]
    )

    # One caption for the whole chunk: the cue runs to the end of the audio (the
    # recognizer only trims a leading silence), so the tail is never lost.
    assert spans == [(480, 24_000)]
