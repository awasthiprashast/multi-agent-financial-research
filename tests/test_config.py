import os

import config


def test_env_file_is_parsed_and_real_env_wins(tmp_path, monkeypatch):
    f = tmp_path / ".env"
    f.write_text('# comment\nFR_TEST_A=from_file\nFR_TEST_B="quoted"\nFR_TEST_EMPTY=\n'
                 "FR_TEST_C = spaced \n", encoding="utf-8")
    monkeypatch.setenv("FR_TEST_A", "from_shell")
    for k in ("FR_TEST_B", "FR_TEST_C", "FR_TEST_EMPTY"):
        monkeypatch.delenv(k, raising=False)
    config.load_env(f)
    assert os.environ["FR_TEST_A"] == "from_shell"      # shell beats file
    assert os.environ["FR_TEST_B"] == "quoted"
    assert os.environ["FR_TEST_C"] == "spaced"
    assert "FR_TEST_EMPTY" not in os.environ            # blank values are ignored
    for k in ("FR_TEST_B", "FR_TEST_C"):
        monkeypatch.delenv(k, raising=False)


def test_missing_env_file_is_fine(tmp_path):
    config.load_env(tmp_path / "nope.env")
