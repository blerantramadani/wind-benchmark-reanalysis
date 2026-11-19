"""The preprint files and archived forecasts are unchanged, and the helpers copied from script 08 are identical."""
import ast
import hashlib
import inspect
import textwrap

import pytest

from wbr import config as C
from wbr.y2018 import common as K


@pytest.mark.parametrize("folder", ["preprint", "archived"])
def test_checksums(folder):
    root = C.ROOT / folder
    lines = (root / "SHA256SUMS").read_text().splitlines()
    assert lines
    listed = set()
    for line in lines:
        digest, name = line.split(maxsplit=1)
        listed.add(name.removeprefix("./"))
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name
    present = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() and p.name != "SHA256SUMS"}
    assert present == listed


def _source_08():
    src = (K.PREPRINT / "scripts" / "08_final_benchmark.py").read_text()
    return src, ast.parse(src)


@pytest.mark.parametrize("name", ["nrmse", "load_data", "valid_origins", "empirical_power_curve"])
def test_helpers_copied_unchanged(name):
    src, tree = _source_08()
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    assert textwrap.dedent(inspect.getsource(getattr(K, name))).strip() == ast.get_source_segment(src, node).strip()


def test_constants_copied_unchanged():
    src, tree = _source_08()
    consts = {t.id: ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
              for t in n.targets if isinstance(t, ast.Name) and t.id in ("DATASETS", "HORIZONS")}
    assert consts == {"DATASETS": K.DATASETS, "HORIZONS": K.HORIZONS}
    arch = (K.PREPRINT / "scripts" / "architectures.py").read_text()
    assert f"LOOK_BACK = {K.LOOK_BACK}" in arch
