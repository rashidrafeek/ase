import numpy as np

from ase import Atoms
from ase.calculators.calculator import BaseCalculator, all_properties


class FiniteDifferenceCalculator(BaseCalculator):
    """Wrapper calculator using the finite-difference method.

    The forces and the stress are computed using the finite-difference method.

    .. versionadded:: 3.24.0
    """

    implemented_properties = all_properties

    def __init__(
        self,
        calc: BaseCalculator,
        dforces: float = 1e-3,
        dstress: float = 1e-6,
    ) -> None:
        """

        Parameters
        ----------
        calc : :class:`~ase.calculators.calculator.BaseCalculator`
            ASE Calculator object to be wrapped.
        dforces : float, default 1e-3
            Step size used for computing forces.
        dstress : float, default 1e-6
            Step size used for computing stress.

        """
        super().__init__()
        self.calc = calc
        self.dforces = dforces
        self.dstress = dstress

    def calculate(self, atoms: Atoms, properties, system_changes) -> None:
        atoms = atoms.copy()  # copy to not mess up original `atoms`
        atoms.calc = self.calc
        self.results = {
            'energy': atoms.get_potential_energy(),
            'forces': calc_numerical_forces(atoms, d=self.dforces),
            'stress': calc_numerical_stress(atoms, d=self.dstress),
        }
        self.results['free_energy'] = self.results['energy']


def _numeric_force(atoms: Atoms, a: int, i: int, d: float = 0.001) -> float:
    """Calculate numerical force on a specific atom along a specific direction.

    Parameters
    ----------
    atoms : :class:`~ase.Atoms`
        ASE :class:`~ase.Atoms` object.
    a : int
        Index of atoms.
    i : {0, 1, 2}
        Index of Cartesian component.
    d : float, default 0.001
        Step size.

    """
    p0 = atoms.get_positions()
    p = p0.copy()
    p[a, i] = p0[a, i] + d
    atoms.set_positions(p, apply_constraint=False)
    eplus = atoms.get_potential_energy()
    p[a, i] = p0[a, i] - d
    atoms.set_positions(p, apply_constraint=False)
    eminus = atoms.get_potential_energy()
    atoms.set_positions(p0, apply_constraint=False)
    return (eminus - eplus) / (2 * d)


def calc_numerical_forces(
    atoms: Atoms,
    d: float = 0.001,
) -> np.ndarray:
    """Calculate forces numerically based on the finite-difference method.

    Parameters
    ----------
    atoms : :class:`~ase.Atoms`
        ASE :class:`~ase.Atoms` object.
    d : float, default 1e-6
        Displacement.

    Returns
    -------
    forces : np.ndarray
        Forces computed numerically based on the finite-difference method.

    """
    return np.array([[_numeric_force(atoms, a, i, d)
                      for i in range(3)] for a in range(len(atoms))])


def calc_numerical_stress(
    atoms: Atoms,
    d: float = 1e-6,
    voigt: bool = True,
) -> np.ndarray:
    """Calculate stress numerically based on the finite-difference method.

    Parameters
    ----------
    atoms : :class:`~ase.Atoms`
        ASE :class:`~ase.Atoms` object.
    d : float, default 1e-6
        Strain in the Voigt notation.
    voigt : bool, default True
        If True, the stress is returned in the Voigt notation.

    Returns
    -------
    stress : np.ndarray
        Stress computed numerically based on the finite-difference method.

    """
    stress = np.zeros((3, 3), dtype=float)

    cell = atoms.cell.copy()
    volume = atoms.get_volume()
    for i in range(3):
        x = np.eye(3)
        x[i, i] = 1.0 + d
        atoms.set_cell(cell @ x, scale_atoms=True)
        eplus = atoms.get_potential_energy(force_consistent=True)

        x[i, i] = 1.0 - d
        atoms.set_cell(cell @ x, scale_atoms=True)
        eminus = atoms.get_potential_energy(force_consistent=True)

        stress[i, i] = (eplus - eminus) / (2 * d * volume)
        x[i, i] = 1.0

        j = i - 2
        x[i, j] = x[j, i] = +0.5 * d
        atoms.set_cell(cell @ x, scale_atoms=True)
        eplus = atoms.get_potential_energy(force_consistent=True)

        x[i, j] = x[j, i] = -0.5 * d
        atoms.set_cell(cell @ x, scale_atoms=True)
        eminus = atoms.get_potential_energy(force_consistent=True)

        stress[i, j] = stress[j, i] = (eplus - eminus) / (2 * d * volume)

    atoms.set_cell(cell, scale_atoms=True)

    return stress.flat[[0, 4, 8, 5, 2, 1]] if voigt else stress
