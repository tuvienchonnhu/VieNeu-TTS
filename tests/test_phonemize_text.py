from unittest.mock import Mock, patch

from vieneu_utils.phonemize_text import (
    PuncNormalizer,
    _phonemize_cached,
    phonemize_batch,
    phonemize_text,
)


def test_punc_normalizer_maps_latin_eth_to_vietnamese_d():
    with patch("vieneu_utils.phonemize_text.Normalizer") as normalizer_cls:
        punc_normalizer = PuncNormalizer()
        punc_normalizer.normalize("ÐƯỜNG TU và ðường tu")
        punc_normalizer.normalize_batch(["ÐƯỜNG TU", "ðường tu"])

    normalizer_cls.return_value.normalize.assert_called_once_with(
        "ĐƯỜNG TU và đường tu", punc_norm=True
    )
    normalizer_cls.return_value.normalize_batch.assert_called_once_with(
        ["ĐƯỜNG TU", "đường tu"], punc_norm=True
    )


def test_phonemize_text_maps_latin_eth_before_pipeline():
    pipeline = Mock()
    pipeline.run.return_value = "expected phones"
    _phonemize_cached.cache_clear()
    try:
        with patch("vieneu_utils.phonemize_text._get_pipeline", return_value=pipeline):
            assert phonemize_text("ÐƯỜNG TU") == "expected phones"
        pipeline.run.assert_called_once_with("ĐƯỜNG TU", punc_norm=True)
    finally:
        _phonemize_cached.cache_clear()


def test_phonemize_text_joins_lau_hoac_phone_boundary():
    pipeline = Mock()
    pipeline.run.return_value = "lˈə6w hwˌa6c."
    _phonemize_cached.cache_clear()
    try:
        with patch("vieneu_utils.phonemize_text._get_pipeline", return_value=pipeline):
            assert phonemize_text("Lậu Hoặc") == "lˈə6whwˌa6c."
    finally:
        _phonemize_cached.cache_clear()


def test_phonemize_batch_maps_latin_eth_when_normalization_is_skipped():
    g2p = Mock()
    g2p.phonemize_batch.return_value = ["phones"]

    with patch("vieneu_utils.phonemize_text._get_g2p", return_value=g2p):
        assert phonemize_batch(["ÐƯỜNG TU"], skip_normalize=True) == ["phones"]

    g2p.phonemize_batch.assert_called_once_with(
        ["ĐƯỜNG TU"], punc_norm=True, phoneme_dict=None
    )


def test_phonemize_batch_joins_lau_hoac_phone_boundary():
    g2p = Mock()
    g2p.phonemize_batch.return_value = ["lˈə6w hwˌa6c."]

    with patch("vieneu_utils.phonemize_text._get_g2p", return_value=g2p):
        assert phonemize_batch(["lậu hoặc"], skip_normalize=True) == [
            "lˈə6whwˌa6c."
        ]
