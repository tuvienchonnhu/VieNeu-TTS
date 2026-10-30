from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np

from apps.ctc_alignment import VietnameseCTCAligner, _ctc_token_spans


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

    assert spans == [(480, 12_000)]
