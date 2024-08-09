import pytest

from ase.build import bulk
from ase.calculators.calculator import compare_atoms
from ase.io.amber import write_amber_coordinates, read_amber_coordinates


def test_io_amber_coordinates():
    atoms = bulk('Au', orthorhombic=True)
    filename = 'amber.netcdf'
    write_amber_coordinates(atoms, filename)

    print(atoms.cell)
    atoms2 = read_amber_coordinates(filename)
    print(atoms2.cell)
    # The format does not save the species so they revert to 'X'
    assert all(atoms2.symbols == 'X')
    assert compare_atoms(atoms, atoms2) == ['numbers']


def test_cannot_write_nonorthorhombic():
    atoms = bulk('Ti')
    with pytest.raises(ValueError, match='Non-orthorhombic'):
        write_amber_coordinates(atoms, 'xxx')
