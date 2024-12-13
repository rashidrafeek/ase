import pytest


@pytest.fixture()
def get_db_name():
    """ Fixture that returns a function to get the test db name
    for the different supported db types.

    Args:
        dbtype (str): Type of database. Currently only 5 types supported:
            postgresql, mysql, mariadb, json, and db (sqlite3)
    """
    def _func(dbtype):
        name = None

        if dbtype == 'json':
            name = 'testase.json'
        elif dbtype == 'db':
            name = 'testase.db'
        else:
            raise ValueError(f'Bad db type: {dbtype}')

        if name is None:
            pytest.skip('Test requires environment variables')

        return name

    return _func
