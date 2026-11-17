from pathlib import Path

from scripts import generate_head, plot_head

CONFIG = Path(__file__).resolve().parents[3] / "configs" / "head_small.yaml"


def test_generate_then_plot(tmp_path, capsys):
    head_path = tmp_path / "head.h5"
    figure_path = tmp_path / "head.png"
    assert (
        generate_head.main(["--config", str(CONFIG), "--seed", "1", "--out", str(head_path)]) == 0
    )
    printed = capsys.readouterr().out
    assert "GiB peak memory" in printed and "saved" in printed
    assert plot_head.main(["--head", str(head_path), "--out", str(figure_path)]) == 0
    assert figure_path.stat().st_size > 10_000
