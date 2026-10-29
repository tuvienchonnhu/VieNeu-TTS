import re

import pytest

from apps.text_subtitles import build_estimated_srt


def test_build_estimated_srt_creates_ordered_sentence_cues():
    subtitle = build_estimated_srt(
        "Câu thứ nhất. Câu thứ hai dài hơn.", duration_seconds=10
    )
    cues = subtitle.strip().split("\n\n")

    assert len(cues) == 2
    assert cues[0].startswith("1\n00:00:00,000 --> ")
    assert "Câu thứ nhất." in cues[0]
    assert cues[1].startswith("2\n")
    assert "Câu thứ hai dài hơn." in cues[1]

    first_end = re.search(r"--> (.+)", cues[0]).group(1)
    second_start = re.search(r"\n(.+) -->", cues[1]).group(1)
    assert first_end == second_start
    assert cues[1].endswith("00:00:10,000\nCâu thứ hai dài hơn.")


def test_build_estimated_srt_preserves_paragraph_and_uppercase_heading():
    subtitle = build_estimated_srt(
        "GIỮ GÌN OAI NGHI TRONG GIỜ HỌC\nNội dung bài học bắt đầu ở đây.",
        duration_seconds=8,
    )

    assert "GIỮ GÌN OAI NGHI TRONG GIỜ HỌC" in subtitle
    assert "Nội dung bài học bắt đầu ở đây." in subtitle
    assert subtitle.count("-->") == 2


@pytest.mark.parametrize(
    ("text", "duration"),
    [("", 2), ("   ", 2), ("Có nội dung.", 0)],
)
def test_build_estimated_srt_rejects_empty_text_or_duration(text, duration):
    with pytest.raises(ValueError):
        build_estimated_srt(text, duration)
