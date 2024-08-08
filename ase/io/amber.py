import numpy as np
import ase.units as units


def write_amber_coordinates(atoms, fout):
    fout.Conventions = 'AMBERRESTART'
    fout.ConventionVersion = "1.0"
    fout.title = 'Ase-generated-amber-restart-file'
    fout.application = "AMBER"
    fout.program = "ASE"
    fout.programVersion = "1.0"
    fout.createDimension('cell_spatial', 3)
    fout.createDimension('label', 5)
    fout.createDimension('cell_angular', 3)
    fout.createDimension('time', 1)
    time = fout.createVariable('time', 'd', ('time',))
    time.units = 'picosecond'
    time[0] = 0
    fout.createDimension('spatial', 3)
    spatial = fout.createVariable('spatial', 'c', ('spatial',))
    spatial[:] = np.asarray(list('xyz'))

    natom = len(atoms)
    fout.createDimension('atom', natom)
    coordinates = fout.createVariable('coordinates', 'd',
                                      ('atom', 'spatial'))
    coordinates.units = 'angstrom'
    coordinates[:] = atoms.get_positions()[:]

    if atoms.get_velocities() is not None:
        velocities = fout.createVariable('velocities', 'd',
                                         ('atom', 'spatial'))
        velocities.units = 'angstrom/picosecond'
        # Amber's units of time are 1/20.455 ps
        # Any other units are ignored in restart files, so these
        # are the only ones it is safe to print
        # See: http://ambermd.org/Questions/units.html
        # Apply conversion factor from ps:
        velocities.scale_factor = 20.455
        # get_velocities call returns velocities with units sqrt(eV/u)
        # so convert to Ang/ps
        factor = units.fs * 1000 / velocities.scale_factor
        velocities[:] = atoms.get_velocities()[:] * factor

    # title
    cell_angular = fout.createVariable('cell_angular', 'c',
                                       ('cell_angular', 'label'))
    cell_angular[0] = np.asarray(list('alpha'))
    cell_angular[1] = np.asarray(list('beta '))
    cell_angular[2] = np.asarray(list('gamma'))

    # title
    cell_spatial = fout.createVariable('cell_spatial', 'c', ('cell_spatial',))
    cell_spatial[0], cell_spatial[1], cell_spatial[2] = 'a', 'b', 'c'

    # data
    cell_lengths = fout.createVariable('cell_lengths', 'd', ('cell_spatial',))
    cell_lengths.units = 'angstrom'
    cell_lengths[0] = atoms.get_cell()[0, 0]
    cell_lengths[1] = atoms.get_cell()[1, 1]
    cell_lengths[2] = atoms.get_cell()[2, 2]

    cell_angles = fout.createVariable('cell_angles', 'd', ('cell_angular',))
    box_alpha, box_beta, box_gamma = 90.0, 90.0, 90.0
    cell_angles[0] = box_alpha
    cell_angles[1] = box_beta
    cell_angles[2] = box_gamma

    cell_angles.units = 'degree'
