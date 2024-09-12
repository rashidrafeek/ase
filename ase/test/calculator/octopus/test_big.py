import pytest

from ase.build import bulk, graphene_nanoribbon
from ase.collections import g2


def calculate(factory, system, **kwargs):
    calc = factory.calc(**kwargs)
    system.calc = calc
    system.get_potential_energy()
    calc.get_eigenvalues()
    return calc


calc = pytest.mark.calculator


@calc('octopus', Spacing='0.2 * angstrom')
def test_o2(factory):
    atoms = g2['O2']
    atoms.center(vacuum=2.5)
    calculate(factory,
              atoms,
              BoxShape='parallelepiped',
              SpinComponents='spin_polarized',
              ExtraStates=2)


@calc('octopus')
def test_si(factory):
    calc = calculate(factory,
                     bulk('Si'),  # , orthorhombic=True),
                     KPointsGrid=[[4, 4, 4]],
                     KPointsUseSymmetries=True,
                     SmearingFunction='fermi_dirac',
                     ExtraStates=2,
                     Smearing='0.1 * eV',
                     ExperimentalFeatures=True,
                     Spacing='0.45 * Angstrom')
    eF = calc.get_fermi_level()
    print('eF', eF)


if 0:
    # Experimental feature: mixed periodicity.  Let us not do this for now...
    graphene = graphene_nanoribbon(2, 2, sheet=True)
    graphene.positions = graphene.positions[:, [0, 2, 1]]
    graphene.pbc = [1, 1, 0]  # from 1, 0, 1
    calc = calculate('graphene',
                     graphene,
                     KPointsGrid=[[2, 1, 2]],
                     KPointsUseSymmetries=True,
                     ExperimentalFeatures=True,
                     ExtraStates=4,
                     SmearingFunction='fermi_dirac',
                     Smearing='0.1 * eV')
