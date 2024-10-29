from ase import Atoms
from ase.calculators.calculator import BaseCalculator, all_properties
from ase.calculators.test import numeric_forces, numeric_stress


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
            'forces': numeric_forces(atoms, d=self.dforces),
            'stress': numeric_stress(atoms, d=self.dstress),
        }
        self.results['free_energy'] = self.results['energy']
