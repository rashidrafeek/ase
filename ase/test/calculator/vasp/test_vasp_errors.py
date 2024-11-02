"""Test module for explicitly unittesting errors generated by VASP calculator"""

import pytest

from ase import Atoms
from ase.build import molecule
from ase.calculators.calculator import CalculationFailed
from ase.calculators.vasp import Vasp


@pytest.fixture(name="atoms")
def fixture_atoms():
    return molecule('H2', vacuum=5, pbc=True)


def test_bad_executable_stderr(atoms, tmp_path, monkeypatch):
    monkeypatch.setenv("VASP_PP_PATH", str(tmp_path))
    (tmp_path / "H").mkdir()
    with open(tmp_path / "H" / "POTCAR", "w") as fout:
        fout.write("\n")
    calc = Vasp(encut=100, command=str(tmp_path / "_NO_VASP_EXEC_"), pp=".", directory=tmp_path)
    atoms.calc = calc
    try:
        atoms.get_potential_energy()
    except CalculationFailed as exc:
        exc_str = str(exc)

    # stderr capture should put stderr in exception text
    assert 'stderr' in exc_str
    # content of stderr should be in exception, and mention path to failed executable
    assert str(tmp_path / '_NO_VASP_EXEC') in exc_str
