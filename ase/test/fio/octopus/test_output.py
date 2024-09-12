"""Tests for Octopus outputs."""
import numpy as np

from ase.io.octopus.output import read_static_info
from ase.units import Debye


def test_dipole_moment(datadir):
    """Test if the dipole moment is parsed correctly."""
    file = datadir / 'octopus/linear_response_02-h2o_pol_lr.01_h2o_gs_info'
    with file.open(encoding='utf-8') as fd:
        results = read_static_info(fd)
    dipole_ref = np.array((7.45151E-16, 9.30594E-01, 3.24621E-15)) * Debye
    np.testing.assert_allclose(results['dipole'], dipole_ref)
