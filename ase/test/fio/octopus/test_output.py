"""Tests for Octopus outputs."""
import numpy as np

from ase.io.octopus.output import read_static_info
from ase.units import Debye, Hartree


def test_fermi_level(datadir):
    """Test if the Fermi level is parsed correctly."""
    file = datadir / 'octopus/periodic_systems_25-Fe_polarized.01-gs_info'
    with file.open(encoding='utf-8') as fd:
        results = read_static_info(fd)
    efermi_ref = 0.153766 * Hartree
    np.testing.assert_allclose(results['fermi_level'], efermi_ref)


def test_dipole_moment(datadir):
    """Test if the dipole moment is parsed correctly."""
    file = datadir / 'octopus/linear_response_02-h2o_pol_lr.01_h2o_gs_info'
    with file.open(encoding='utf-8') as fd:
        results = read_static_info(fd)
    dipole_ref = np.array((7.45151E-16, 9.30594E-01, 3.24621E-15)) * Debye
    np.testing.assert_allclose(results['dipole'], dipole_ref)


def test_magnetic_moment(datadir):
    """Test if the magnetic moment is parsed correctly."""
    file = datadir / 'octopus/periodic_systems_25-Fe_polarized.01-gs_info'
    with file.open(encoding='utf-8') as fd:
        results = read_static_info(fd)
    magmom_ref = 7.409638
    magmoms_ref = [3.385730, 3.385730]
    np.testing.assert_allclose(results['magmom'], magmom_ref)
    np.testing.assert_allclose(results['magmoms'], magmoms_ref)
