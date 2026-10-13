import json

from clipforge.config import Config
from clipforge.transcribe import Transcript, Utterance, Word, _cache_file


def make_transcript() -> Transcript:
    words = [
        Word(0.0, 0.5, "nadie"),
        Word(0.5, 0.9, "se"),
        Word(0.9, 1.4, "fija"),
        Word(3.0, 3.4, "en"),
        Word(3.4, 3.9, "su"),
        Word(3.9, 4.6, "mano"),
        Word(10.0, 10.4, "fuera"),
    ]
    utts = [Utterance(0.0, 1.4, "nadie se fija"), Utterance(3.0, 4.6, "en su mano")]
    return Transcript(language="es", duration_s=12.0, model="small", words=words, utterances=utts)


def test_words_between_uses_midpoints():
    t = make_transcript()
    expected = ["nadie", "se", "fija", "en", "su", "mano"]
    assert [w.text for w in t.words_between(0.0, 5.0)] == expected
    assert [w.text for w in t.words_between(3.0, 5.0)] == ["en", "su", "mano"]


def test_text_between():
    assert make_transcript().text_between(3.0, 5.0) == "en su mano"


def test_dialogue_density_and_speech_ratio():
    t = make_transcript()
    assert t.dialogue_density(0.0, 5.0) == 6 / 5.0
    # 1.4 s of speech in the first window + 1.6 s in the second = 3.0 s over 5 s
    assert abs(t.speech_ratio(0.0, 5.0) - 0.6) < 1e-6
    assert t.speech_ratio(5.0, 9.0) == 0.0


def test_utterances_between_is_overlap_based():
    t = make_transcript()
    assert len(t.utterances_between(1.0, 3.5)) == 2
    assert t.utterances_between(5.0, 9.0) == []


def test_roundtrip_serialisation():
    t = make_transcript()
    back = Transcript.from_dict(json.loads(json.dumps(t.to_dict())))
    assert back.words == t.words
    assert back.utterances == t.utterances
    assert back.language == "es"


def test_cache_key_includes_model_and_language(tmp_path):
    cfg = Config(project_root=tmp_path)
    a = _cache_file(cfg, "abc", "small", "es")
    b = _cache_file(cfg, "abc", "medium", "es")
    c = _cache_file(cfg, "abc", "small", None)
    assert a != b != c and a != c
    assert "auto" in c.name


def test_empty_transcript_is_safe():
    t = Transcript()
    assert t.text_between(0, 100) == ""
    assert t.dialogue_density(0, 100) == 0.0
    assert t.speech_ratio(0, 100) == 0.0
