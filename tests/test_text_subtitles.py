import re

import numpy as np
import pytest

from apps.text_subtitles import build_aligned_srt, write_aligned_srt


def test_srt_timestamps_follow_exact_audio_chunk_lengths_and_gaps():
    subtitle = build_aligned_srt(
        ["Câu thứ nhất.", "Câu thứ hai dài hơn."],
        [np.ones(1_000, dtype=np.float32), np.ones(2_000, dtype=np.float32)],
        sample_rate=1_000,
        gap_samples=[500],
    )
    cues = subtitle.strip().split("\n\n")

    assert len(cues) == 2
    assert cues[0] == "1\n00:00:00,000 --> 00:00:01,000\nCâu thứ nhất."
    assert cues[1] == "2\n00:00:01,500 --> 00:00:03,500\nCâu thứ hai dài hơn."


def test_srt_uses_actual_padding_for_minimum_pause_join():
    audio = [
        np.ones(1_000, dtype=np.float32),
        np.ones(2_000, dtype=np.float32),
    ]
    subtitle = build_aligned_srt(
        ["GIỮ GÌN OAI NGHI TRONG GIỜ HỌC", "Nội dung bài học bắt đầu ở đây."],
        audio,
        sample_rate=1_000,
        silence_ps=[0.5],
    )

    cues = subtitle.strip().split("\n\n")
    assert cues[0].endswith("00:00:01,000\nGIỮ GÌN OAI NGHI TRONG GIỜ HỌC")
    assert cues[1].startswith("2\n00:00:01,500 --> 00:00:03,500\n")


def test_multiple_sentences_split_within_chunk_duration():
    subtitle = build_aligned_srt(
        ["Câu ngắn. Câu dài hơn một chút."],
        [np.ones(2_000, dtype=np.float32)],
        sample_rate=1_000,
    )
    cues = subtitle.strip().split("\n\n")

    assert len(cues) == 2
    first_end = re.search(r"--> (.+)", cues[0]).group(1)
    second_start = re.search(r"\n(.+) -->", cues[1]).group(1)
    assert first_end == second_start
    assert cues[0].startswith("1\n00:00:00,000 --> ")
    assert cues[1].endswith("00:00:02,000\nCâu dài hơn một chút.")


def test_ctc_alignment_uses_acoustic_sentence_boundaries():
    class FakeAligner:
        def align_captions(self, audio, sample_rate, captions):
            if len(captions) == 2:
                assert captions == ["Câu ngắn.", "Câu dài hơn một chút."]
                return [(100, 500), (1_100, 1_900)]
            assert captions == ["Đoạn tiếp."]
            return [(200, 800)]

    subtitle = build_aligned_srt(
        ["Câu ngắn. Câu dài hơn một chút.", "Đoạn tiếp."],
        [np.ones(2_000, dtype=np.float32), np.ones(1_000, dtype=np.float32)],
        sample_rate=1_000,
        gap_samples=[300],
        aligner=FakeAligner(),
    )
    cues = subtitle.strip().split("\n\n")

    assert cues[0] == "1\n00:00:00,100 --> 00:00:00,500\nCâu ngắn."
    assert cues[1] == "2\n00:00:01,100 --> 00:00:01,900\nCâu dài hơn một chút."
    assert cues[2] == "3\n00:00:02,500 --> 00:00:03,100\nĐoạn tiếp."


def test_ctc_alignment_uses_display_sentences_when_normalization_changes_punctuation():
    class FakeAligner:
        def align_captions(self, audio, sample_rate, captions):
            assert captions == ["Câu thứ nhất.", "Câu thứ hai."]
            return [(100, 800), (900, 1_900)]

    warnings = []
    subtitle = build_aligned_srt(
        ["Câu thứ nhất câu thứ hai"],
        [np.ones(2_000, dtype=np.float32)],
        sample_rate=1_000,
        display_chunks=["Câu thứ nhất. Câu thứ hai."],
        aligner=FakeAligner(),
        alignment_warnings=warnings,
    )

    cues = subtitle.strip().split("\n\n")
    assert cues[0] == "1\n00:00:00,100 --> 00:00:00,800\nCâu thứ nhất."
    assert cues[1] == "2\n00:00:00,900 --> 00:00:01,900\nCâu thứ hai."
    assert warnings == []


def test_failed_ctc_alignment_reports_character_estimate_fallback():
    class FailedAligner:
        def align_captions(self, audio, sample_rate, captions):
            raise ValueError("không căn chỉnh được")

    warnings = []
    build_aligned_srt(
        ["Một câu. Câu tiếp theo."],
        [np.ones(2_000, dtype=np.float32)],
        sample_rate=1_000,
        aligner=FailedAligner(),
        alignment_warnings=warnings,
    )

    assert len(warnings) == 1
    assert "không căn chỉnh được" in warnings[0]
    assert "ước lượng theo ký tự" in warnings[0]


def test_srt_rejects_mismatched_chunks_and_audio():
    with pytest.raises(ValueError, match="khớp"):
        build_aligned_srt(["Câu."], [], sample_rate=24_000)


def test_srt_rejects_invalid_pause_count():
    with pytest.raises(ValueError, match="khoảng nghỉ"):
        build_aligned_srt(
            ["Một câu.", "Câu nữa."],
            [np.ones(100), np.ones(100)],
            sample_rate=100,
            gap_samples=[],
        )


def test_write_aligned_srt_creates_sidecar_next_to_audio(tmp_path):
    audio_path = tmp_path / "speech.wav"
    audio_path.touch()

    subtitle_path = write_aligned_srt(
        str(audio_path),
        ["Một câu."],
        [np.ones(1_000, dtype=np.float32)],
        sample_rate=1_000,
    )

    assert subtitle_path == str(tmp_path / "speech.srt")
    assert "00:00:01,000" in (tmp_path / "speech.srt").read_text(encoding="utf-8-sig")


def test_write_aligned_srt_preserves_original_case_and_spelling(tmp_path):
    audio_path = tmp_path / "speech.wav"
    audio_path.touch()

    subtitle_path = write_aligned_srt(
        str(audio_path),
        ["giữ gìn oai nghi trong giờ học.", "các em chú ý nghe giảng."],
        [np.ones(1_000, dtype=np.float32), np.ones(2_000, dtype=np.float32)],
        sample_rate=1_000,
        source_text="GIỮ GÌN OAI NGHI TRONG GIỜ HỌC.\nCác em chú ý nghe giảng.",
        gap_samples=[500],
    )

    subtitle = (tmp_path / "speech.srt").read_text(encoding="utf-8-sig")
    assert "GIỮ GÌN OAI NGHI TRONG GIỜ HỌC." in subtitle
    assert "Các em chú ý nghe giảng." in subtitle
    assert "giữ gìn oai nghi" not in subtitle
    assert subtitle_path == str(tmp_path / "speech.srt")
