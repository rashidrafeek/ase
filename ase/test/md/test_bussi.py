import numpy as np
import pytest

from ase import units
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.md.bussi import Bussi
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution


def test_bussi():
    atoms = bulk("Pt") * (10, 10, 10)
    atoms.calc = EMT()

    with pytest.raises(ValueError):
        Bussi(atoms, 0.1 * units.fs, 300, 100 * units.fs)

    MaxwellBoltzmannDistribution(atoms, temperature_K=300)

    dyn = Bussi(atoms, 0.1 * units.fs, 300, 100 * units.fs)

    dyn.run(10)


def test_bussi_transfered_energy_conservation():
    atoms = bulk("Cu") * (5, 5, 5)
    atoms.calc = EMT()

    MaxwellBoltzmannDistribution(
        atoms, temperature_K=300, rng=np.random.default_rng(seed=42)
    )

    dyn = Bussi(
        atoms,
        1.0e-5 * units.fs,
        300,
        100 * units.fs,
        rng=np.random.default_rng(seed=42),
    )

    conserved_quantity = []

    for _ in dyn.irun(100):
        conserved_quantity.append(
            dyn.atoms.get_total_energy() - dyn.transferred_energy
        )

    assert np.unique(np.round(conserved_quantity, 10)).size == 1


def test_bussi_paranoia_check():
    atoms = bulk("Cu") * (3, 3, 3)
    atoms.calc = EMT()

    MaxwellBoltzmannDistribution(
        atoms,
        temperature_K=300,
        rng=np.random.default_rng(seed=10),
        force_temp=True,
    )

    dyn = Bussi(
        atoms,
        1.0e-100 * units.fs,
        300,
        1.0e-100 * units.fs,
        rng=np.random.default_rng(seed=10),
    )

    temperatures = []

    for _ in dyn.irun(1000):
        temperatures.append(dyn.atoms.get_temperature())

    assert np.mean(temperatures) == pytest.approx(300, abs=5.0)
