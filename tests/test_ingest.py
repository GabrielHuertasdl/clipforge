import pytest

from clipforge.config import Config, load_config
from clipforge.ingest import extract_year, file_hash, parse_title


@pytest.fixture(scope="module")
def cfg() -> Config:
    return load_config("config.yaml")


@pytest.mark.parametrize(
    "filename,title,year",
    [
        ("Heat.1995.1080p.BluRay.x264-GROUP.mkv", "Heat", 1995),
        ("El.Padrino.1972.REMASTERED.1080p.BluRay.DTS.x264.mkv", "El Padrino", 1972),
        ("Blade_Runner_2049_(2017)_2160p_UHD_HDR.mp4", "Blade Runner 2049", 2017),
        ("The Matrix [1999] 720p BRRip [Dual][Latino].avi", "The Matrix", 1999),
        ("Interstellar.2014.EXTENDED.1080p.WEB-DL.DDP5.1.H264.mkv", "Interstellar", 2014),
        ("mad max fury road 2015 1080p.mkv", "Mad Max Fury Road", 2015),
        ("SEVEN.1995.1080P.BLURAY.X264.MKV", "Seven", 1995),
    ],
)
def test_parse_title(cfg, filename, title, year):
    got_title, got_year = parse_title(filename, cfg)
    assert (got_title, got_year) == (title, year)


def test_leading_year_stays_in_title(cfg):
    # "2001: A Space Odyssey" — the leading number is the title, the trailing one the year.
    title, year = parse_title("2001.A.Space.Odyssey.1968.1080p.BluRay.x264.mkv", cfg)
    assert title == "2001 A Space Odyssey"
    assert year == 1968


def test_no_year(cfg):
    title, year = parse_title("Amanece.Que.No.Es.Poco.1080p.mkv", cfg)
    assert title == "Amanece Que No Es Poco"
    assert year is None


def test_title_never_empty(cfg):
    title, _ = parse_title("1080p.x264.mkv", cfg)
    assert title


def test_extract_year_keeps_rest():
    year, rest = extract_year("Heat 1995 1080p", r"(19\d{2}|20\d{2})")
    assert year == 1995
    assert "1995" not in rest


def test_file_hash_partial_is_deterministic_and_content_sensitive(tmp_path):
    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"
    a.write_bytes(b"x" * 1000)
    b.write_bytes(b"y" * 1000)
    assert file_hash(a, chunk_bytes=64) == file_hash(a, chunk_bytes=64)
    assert file_hash(a, chunk_bytes=64) != file_hash(b, chunk_bytes=64)


def test_file_hash_full_matches_partial_for_small_files(tmp_path):
    p = tmp_path / "small.bin"
    p.write_bytes(b"z" * 100)
    assert file_hash(p, chunk_bytes=4096) == file_hash(p, chunk_bytes=4096, full=True)
