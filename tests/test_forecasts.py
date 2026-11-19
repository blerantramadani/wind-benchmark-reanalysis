"""Restart rules of the forecast step."""
import json

import pandas as pd

from wbr import config as C
from wbr import forecasts as F


def _cell(tmp_path, mlp, protocol=None):
    for part in ("val", "test"):
        pd.DataFrame({"origin": [1]}).to_csv(tmp_path / f"kelmarsh_2019_normal_h1_{part}.csv", index=False)
    rec = {"protocol_sha256": protocol or F.protocol_hash(), "mlp": mlp}
    (tmp_path / "kelmarsh_2019_normal_h1_done.json").write_text(json.dumps(rec))


def test_restart_rules(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "FORECASTS", tmp_path)
    assert not F.cell_is_done("kelmarsh", 2019, "normal", 1, True)
    _cell(tmp_path, mlp=False)
    assert F.cell_is_done("kelmarsh", 2019, "normal", 1, False)
    assert not F.cell_is_done("kelmarsh", 2019, "normal", 1, True)
    _cell(tmp_path, mlp=True)
    assert F.cell_is_done("kelmarsh", 2019, "normal", 1, True)
    assert F.cell_is_done("kelmarsh", 2019, "normal", 1, False)  # --no-mlp keeps MLP files
    _cell(tmp_path, mlp=True, protocol="0" * 64)
    assert not F.cell_is_done("kelmarsh", 2019, "normal", 1, True)
    (tmp_path / "kelmarsh_2019_normal_h1_val.csv").unlink()
    _cell_rec = tmp_path / "kelmarsh_2019_normal_h1_done.json"
    _cell_rec.write_text(json.dumps({"protocol_sha256": F.protocol_hash(), "mlp": True}))
    assert not F.cell_is_done("kelmarsh", 2019, "normal", 1, True)
