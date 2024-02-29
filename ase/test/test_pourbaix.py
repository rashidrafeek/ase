"""Test Pourbaix diagram."""
import numpy as np
import pytest

from ase.phasediagram import Pourbaix, solvated
from ase.pourbaix import Pourbaix as Pourbaix_new


def test_pourbaix():
    """Test ZnO system from docs."""
    refs = solvated('Zn')
    print(refs)
    refs += [('Zn', 0.0), ('ZnO', -3.323), ('ZnO2(aq)', -2.921)]
    pb = Pourbaix(refs, formula='ZnO')

    _, e = pb.decompose(-1.0, 7.0)
    assert e == pytest.approx(-3.625, abs=0.001)

    U = np.linspace(-2, 2, 3)
    pH = np.linspace(6, 16, 11)
    d, names, text = pb.diagram(U, pH, plot=False)
    print(d, names, text)
    assert d.shape == (3, 11)
    assert d.ptp() == 6
    assert names == ['Zn', 'ZnO2(aq)', 'Zn++(aq)', 'HZnO2-(aq)',
                     'ZnOH+(aq)', 'ZnO', 'ZnO2--(aq)']


def test_new_pourbaix():
    """Test module against Zn Pourbaix diagram from the Atlas"""

    refs = {
        'Zn': 0.0,
        'ZnO': -3.336021896,
        'Zn++(aq)': -1.525613424,
        'ZnOH+(aq)': -3.4125107,
        'HZnO2-(aq)': -4.8087349,
        'ZnO2--(aq)': -4.03387383
    }

    U = np.linspace(-2, 2, 5)
    pH = np.linspace(0, 14, 8)
    pbx = Pourbaix_new('Zn', refs)
    phases, diagram, text, _ = pbx.get_diagrams(U, pH)

    # Verify that the stability domains are the expected ones
    names = [txt[2][0] for txt in text]
    for name in ['Zn', 'ZnO', 'Zn++(aq)', 'HZnO2-(aq)', 'ZnO2--(aq)']:
        assert name in names
    assert 'ZnOH+(aq)' not in names

    # Verify that Zn is stable at U=-2, pH=0
    assert diagram[0, 0] <= 0

    # Verify that Zn++ is the stable phase at U=0, pH=6
    i0 = int(phases[2, 3])
    phase0 = pbx.phases[i0]
    assert 'Zn++(aq)' in phase0.species

    # Verify that the pourbaix energy at U=1, pH=7 is the expected one
    Epbx = pbx.get_pourbaix_energy(1.0, 7.0, verbose=False)[0]
    assert Epbx == pytest.approx(3.880, abs=0.001)
