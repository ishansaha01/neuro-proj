import json

import pytest

from neurosim.common import evalkit
from neurosim.common.evalkit import EvalSpec, eval_case, report
from neurosim.common.io import git_commit

SPEC_TEXT = {
    "claim": "Water is 0 HU",
    "reference": "Definition of the Hounsfield scale",
    "metric": "Mean HU in a water disc [HU]",
    "threshold": "|HU| < 1: below the noise of the reference scan",
}


@pytest.fixture(autouse=True)
def isolated_registry(monkeypatch, tmp_path):
    """Give each test an empty registry and a temporary results directory."""
    monkeypatch.setattr(evalkit, "REGISTRY", {})
    monkeypatch.setattr(evalkit, "_ORIGINS", {})
    monkeypatch.setattr(evalkit, "RESULTS_DIR", tmp_path / "results")


def test_eval_case_registers_the_spec_and_marks_the_function():
    def check():
        return "ran"

    decorated = eval_case(id="E-CT-004", seeds=(0, 3), **SPEC_TEXT)(check)

    assert decorated is check
    assert decorated() == "ran"
    assert [mark.name for mark in decorated.pytestmark] == ["eval"]
    assert evalkit.REGISTRY == {
        "E-CT-004": EvalSpec(id="E-CT-004", seeds=(0, 3), area="ct", **SPEC_TEXT)
    }


def test_default_seed_and_every_area_prefix():
    for number, area in enumerate(evalkit.AREAS):
        eval_case(id=f"E-{area}-{number:03d}", **SPEC_TEXT)(lambda: None)
    assert [spec.area for spec in evalkit.REGISTRY.values()] == [
        "head",
        "fus",
        "ct",
        "data",
        "recon",
        "common",
    ]
    assert all(spec.seeds == (0,) for spec in evalkit.REGISTRY.values())


@pytest.mark.parametrize(
    "bad_id",
    ["E-XRAY-001", "E-FUS-1", "E-FUS-0001", "e-fus-001", "E-FUS-001b", "FUS-001", "E_FUS_001", ""],
)
def test_malformed_id_is_rejected(bad_id):
    with pytest.raises(ValueError, match="does not match"):
        eval_case(id=bad_id, **SPEC_TEXT)
    assert evalkit.REGISTRY == {}


def test_duplicate_id_is_rejected():
    eval_case(id="E-FUS-001", **SPEC_TEXT)(lambda: None)
    with pytest.raises(ValueError, match="duplicate eval id E-FUS-001"):
        eval_case(id="E-FUS-001", **SPEC_TEXT)(lambda: None)
    assert len(evalkit.REGISTRY) == 1


def test_empty_text_and_bad_seeds_are_rejected():
    with pytest.raises(ValueError, match="threshold must not be empty"):
        eval_case(id="E-FUS-001", **{**SPEC_TEXT, "threshold": "  "})
    with pytest.raises(ValueError, match="seeds"):
        eval_case(id="E-FUS-001", seeds=(), **SPEC_TEXT)
    with pytest.raises(ValueError, match="seeds"):
        eval_case(id="E-FUS-001", seeds=(0.5,), **SPEC_TEXT)


def test_report_writes_the_result_file():
    eval_case(id="E-CT-004", **SPEC_TEXT)(lambda: None)
    report("E-CT-004", 0.25, passed=True, note="120 kVp")

    written = (evalkit.RESULTS_DIR / "E-CT-004.json").read_bytes()
    result = json.loads(written.decode("utf-8"))
    assert result == {
        "id": "E-CT-004",
        "value": 0.25,
        "passed": True,
        "note": "120 kVp",
        "git_commit": git_commit(),
    }
    # The same bytes on every operating system: UTF-8, "\n" line ends.
    assert written.endswith(b"}\n") and b"\r" not in written


def test_results_dir_follows_the_variable_then_the_checkout_then_the_working_directory(
    monkeypatch, tmp_path
):
    checkout = tmp_path / "checkout"
    (checkout / "evals").mkdir(parents=True)
    site_packages = tmp_path / "venv" / "lib"
    site_packages.mkdir(parents=True)
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)

    monkeypatch.delenv(evalkit.RESULTS_DIR_VARIABLE, raising=False)
    assert evalkit.resolve_results_dir(checkout) == checkout / "evals" / "results"
    # A non-editable install: no evals/ folder next to the package.
    assert evalkit.resolve_results_dir(site_packages) == work / "evals" / "results"

    monkeypatch.setenv(evalkit.RESULTS_DIR_VARIABLE, str(tmp_path / "chosen"))
    assert evalkit.resolve_results_dir(checkout) == tmp_path / "chosen"
    assert evalkit.resolve_results_dir(site_packages) == tmp_path / "chosen"
    monkeypatch.setenv(evalkit.RESULTS_DIR_VARIABLE, "")
    assert evalkit.resolve_results_dir(checkout) == checkout / "evals" / "results"


def test_report_accepts_several_named_values():
    eval_case(id="E-CT-004", **SPEC_TEXT)(lambda: None)
    report("E-CT-004", {"seed_0": 0.25, "seed_1": 1}, passed=True)
    result = json.loads((evalkit.RESULTS_DIR / "E-CT-004.json").read_text())
    assert result["value"] == {"seed_0": 0.25, "seed_1": 1.0}


def test_failing_report_is_recorded_and_then_fails_the_test():
    eval_case(id="E-CT-004", **SPEC_TEXT)(lambda: None)
    with pytest.raises(AssertionError) as error:
        report("E-CT-004", 3.5, passed=False, note="beam hardening")
    message = str(error.value)
    assert "E-CT-004 failed" in message
    assert "3.5" in message
    assert SPEC_TEXT["threshold"] in message
    assert "beam hardening" in message
    result = json.loads((evalkit.RESULTS_DIR / "E-CT-004.json").read_text())
    assert result["passed"] is False


def test_report_for_an_unregistered_eval_raises():
    with pytest.raises(KeyError):
        report("E-CT-999", 0.0, passed=True)
    assert not evalkit.RESULTS_DIR.exists()
