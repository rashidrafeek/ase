from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

import ase.units
from ase import Atoms
from ase.md.md import MolecularDynamics

if TYPE_CHECKING:
    from typing import IO


# Coefficients for the fourth-order Suzuki-Yoshida integration scheme
# Ref: H. Yoshida, Phys. Lett. A 150, 5-7, 262-268 (1990). https://doi.org/10.1016/0375-9601(90)90092-3
FOURTH_ORDER_COEFFS = [
    1 / (2 - 2 ** (1 / 3)),
    -(2 ** (1 / 3)) / (2 - 2 ** (1 / 3)),
    1 / (2 - 2 ** (1 / 3)),
]


class NoseHooverChainNVT(MolecularDynamics):
    """Isothermal molecular dynamics with Nose-Hoover chain.

    This implementation is based on the Nose-Hoover chain equations and
    the Liouville-operator derived integrator for non-Hamiltonian systems [1-3].

    - [1] G. J. Martyna, M. L. Klein, and M. E. Tuckerman, J. Chem. Phys. 97,
          2635 (1992). https://doi.org/10.1063/1.463940
    - [2] M. E. Tuckerman, J. Alejandre, R. López-Rendón, A. L. Jochim,
          and G. J. Martyna, J. Phys. A: Math. Gen. 39, 5629 (2006).
          https://doi.org/10.1088/0305-4470/39/19/S18
    - [3] M. E. Tuckerman, Statistical Mechanics: Theory and Molecular
          Simulation, Oxford University Press (2010).
    """

    def __init__(
        self,
        atoms: Atoms,
        timestep: float,
        temperature_K: float,
        tdamp: float,
        tchain: int = 3,
        tloop: int = 1,
        trajectory: str | None = None,
        logfile: IO | str | None = None,
        loginterval: int = 1,
    ):
        """
        Parameters
        ----------
        atoms: ase.Atoms
            The atoms object.
        timestep: float
            The time step in ASE time units.
        temperature_K: float
            The target temperature in K.
        tdamp: float
            The characteristic time scale for the thermostat in ASE time units.
            Typically, it is set to 100 times of `timestep`.
        tchain: int
            The number of thermostat variables in the Nose-Hoover chain.
        tloop: int
            The number of sub-steps in thermostat integration.
        trajectory: str or None
            If `trajectory` is str, `Trajectory` will be instantiated.
            Set `None` for no trajectory.
        logfile: IO or str or None
            If `logfile` is str, a file with that name will be opened.
            Set `-` to output into stdout.
        loginterval: int
            Write a log line for every `loginterval` time steps.
        """
        super().__init__(
            atoms=atoms,
            timestep=timestep,
            trajectory=trajectory,
            logfile=logfile,
            loginterval=loginterval,
        )
        assert self.masses.shape == (len(self.atoms), 1)

        self._thermostat = NoseHooverChainThermostat(
            num_atoms=len(self.atoms),
            masses=self.masses,
            temperature_K=temperature_K,
            tdamp=tdamp,
            tchain=tchain,
            tloop=tloop,
        )

        # The following variables are updated during self.step()
        self._q = self.atoms.get_positions()
        self._p = self.atoms.get_momenta()

    def step(self) -> None:
        dt2 = self.dt / 2
        self._p = self._thermostat.integrate_nhc(self._p, dt2)
        self._integrate_p(dt2)
        self._integrate_q(self.dt)
        self._integrate_p(dt2)
        self._p = self._thermostat.integrate_nhc(self._p, dt2)

        self._update_atoms()

    def get_conserved_energy(self) -> float:
        """Return the conserved energy-like quantity.

        This method is mainly used for testing.
        """
        conserved_energy = (
            self.atoms.get_total_energy()
            + self._thermostat.get_thermostat_energy()
        )
        return float(conserved_energy)

    def _update_atoms(self) -> None:
        self.atoms.set_positions(self._q)
        self.atoms.set_momenta(self._p)

    def _get_forces(self) -> np.ndarray:
        self._update_atoms()
        return self.atoms.get_forces(md=True)

    def _integrate_q(self, delta: float) -> None:
        """Integrate exp(i * L_1 * delta)"""
        self._q += delta * self._p / self.masses

    def _integrate_p(self, delta: float) -> None:
        """Integrate exp(i * L_2 * delta)"""
        forces = self._get_forces()
        self._p += delta * forces


class NoseHooverChainThermostat:
    """Nose-Hoover chain style thermostats.

    See `NoseHooverChainNVT` for the references.
    """
    def __init__(
        self,
        num_atoms: int,
        masses: np.ndarray,
        temperature_K: float,
        tdamp: float,
        tchain: int = 3,
        tloop: int = 1,
    ):
        """See `NoseHooverChainNVT` for the parameters."""
        self._num_atoms = num_atoms
        self._masses = masses  # (num_atoms, 1)
        self._tdamp = tdamp
        self._tchain = tchain
        self._tloop = tloop

        self._kT = ase.units.kB * temperature_K

        assert tchain >= 1
        self._Q = np.zeros(tchain)
        self._Q[0] = 3 * num_atoms * self._kT * tdamp**2
        self._Q[1:] = self._kT * tdamp**2

        # The following variables are updated during self.step()
        self._eta = np.zeros(self._tchain)
        self._p_eta = np.zeros(self._tchain)

    def get_thermostat_energy(self) -> float:
        """Return energy-like contribution from the thermostat variables."""
        energy = (
            3 * self._num_atoms * self._kT * self._eta[0]
            + self._kT * np.sum(self._eta[1:])
            + np.sum(0.5 * self._p_eta**2 / self._Q)
        )
        return float(energy)

    def integrate_nhc(self, p: np.ndarray, delta: float) -> np.ndarray:
        """Integrate exp(i * L_NHC * delta) and update momenta `p`."""
        for _ in range(self._tloop):
            for coeff in FOURTH_ORDER_COEFFS:
                p = self._integrate_nhc_loop(
                    p, coeff * delta / len(FOURTH_ORDER_COEFFS)
                )

        return p

    def _integrate_nhc_loop(self, p: np.ndarray, delta: float) -> np.ndarray:
        delta2 = delta / 2
        delta4 = delta / 4

        def _integrate_p_eta_j(p: np.ndarray, j: int) -> None:
            if j < self._tchain - 1:
                self._p_eta[j] *= np.exp(
                    -delta4 * self._p_eta[j + 1] / self._Q[j + 1]
                )

            if j == 0:
                g_j = np.sum(p**2 / self._masses) \
                    - 3 * self._num_atoms * self._kT
            else:
                g_j = self._p_eta[j - 1] ** 2 / self._Q[j - 1] - self._kT
            self._p_eta[j] += delta2 * g_j

            if j < self._tchain - 1:
                self._p_eta[j] *= np.exp(
                    -delta4 * self._p_eta[j + 1] / self._Q[j + 1]
                )

        def _integrate_eta() -> None:
            for j in range(self._tchain):
                self._eta[j] += delta * self._p_eta[j] / self._Q[j]

        def _integrate_nhc_p(p: np.ndarray) -> np.ndarray:
            p *= np.exp(-delta * self._p_eta[0] / self._Q[0])
            return p

        for j in range(self._tchain):
            _integrate_p_eta_j(p, self._tchain - j - 1)
        _integrate_eta()
        p = _integrate_nhc_p(p)
        for j in range(self._tchain):
            _integrate_p_eta_j(p, j)

        return p
