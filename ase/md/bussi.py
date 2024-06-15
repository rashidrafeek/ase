"""Bussi NVT dynamics class."""

import math

import numpy as np

from ase import units
from ase.md.md import MolecularDynamics
from ase.parallel import world


class Bussi(MolecularDynamics):
    """Bussi stochastic velocity rescaling (NVT) molecular dynamics.
    Based on the paper from Bussi et al. (https://arxiv.org/abs/0803.4060)

    Parameters
    ----------
    atoms: Atoms object
        The atoms object.
    timestep: float
        The time step in ASE time units.
    temperature_K: float
        The desired temperature, in Kelvin.
    taut: float
        Time constant for Bussi temperature coupling in ASE time units.
    fix_com: bool (default True)
        If True, the center-of-mass momentum is fixed during the simulation.
    rng: numpy.random.Generator (default np.random.default_rng())
        Random number generator.
    md_kwargs: dict
        Additional arguments passed to MolecularDynamics base class.
    """

    def __init__(
        self,
        atoms,
        timestep,
        temperature_K,
        taut,
        fix_com=True,
        rng=np.random.default_rng(),
        **md_kwargs,
    ):
        super().__init__(
            atoms,
            timestep,
            **md_kwargs,
        )

        self.taut = taut
        self.temp = temperature_K * units.kB
        self.fix_com = fix_com
        self.communicator = world
        self.rng = rng

        self.ndof = self.atoms.get_number_of_degrees_of_freedom()

        if fix_com:
            self.ndof -= 3

        self.target_kinetic_energy = 0.5 * self.temp * self.ndof

        if np.isclose(
            self.atoms.get_kinetic_energy(), 0.0, rtol=0, atol=1e-12
        ):
            raise ValueError("Initial kinetic energy is zero."
                             "Please set initial velocities.")

        self.transferred_energy = 0.0

    def scale_velocities(self):
        """Do the NVT Bussi stochastic velocity scaling."""
        kinetic_energy = self.atoms.get_kinetic_energy()
        alpha = self.calculate_alpha(kinetic_energy)

        momenta = self.atoms.get_momenta()
        self.atoms.set_momenta(alpha * momenta)

        self.transferred_energy += (alpha**2 - 1) * kinetic_energy

    def calculate_alpha(self, kinetic_energy):
        """Calculate the scaling factor alpha using equation (A7)
        from the Bussi paper."""
        exp_term = math.exp(-self.dt / self.taut)
        energy_scaling_term = (
            (1 - exp_term)
            * self.target_kinetic_energy
            / kinetic_energy
            / self.ndof
        )

        r1 = self.rng.standard_normal()
        r2 = self.sum_noises(self.ndof - 1)

        return np.sqrt(
            exp_term
            + energy_scaling_term * (r2 + r1**2)
            + 2 * r1 * np.sqrt(exp_term * energy_scaling_term)
        )

    def sum_noises(self, nn):
        """Sum of nn noises."""
        if nn == 0:
            return 0.0
        elif nn == 1:
            return self.rng.standard_normal() ** 2
        elif nn % 2 == 0:
            return 2.0 * self.rng.standard_gamma(nn / 2)
        else:
            rr = self.rng.standard_normal()
            return 2.0 * self.rng.standard_gamma((nn - 1) / 2) + rr**2

    def step(self, forces=None):
        """Move one timestep forward using Bussi NVT molecular dynamics."""
        self.scale_velocities()

        atoms = self.atoms

        if forces is None:
            forces = atoms.get_forces(md=True)

        momenta = self.atoms.get_momenta()
        momenta += 0.5 * self.dt * forces

        if self.fix_com:
            momenta -= momenta.mean(axis=0)

        self.atoms.set_positions(
            self.atoms.positions
            + self.dt * momenta / self.atoms.get_masses()[:, np.newaxis]
        )

        self.atoms.set_momenta(momenta)
        forces = self.atoms.get_forces(md=True)
        self.atoms.set_momenta(
            self.atoms.get_momenta() + 0.5 * self.dt * forces
        )

        return forces
