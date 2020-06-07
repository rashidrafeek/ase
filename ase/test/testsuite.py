import os
import sys
from subprocess import Popen, PIPE
from contextlib import contextmanager
import importlib
from pathlib import Path
import warnings
import argparse
from multiprocessing import cpu_count

from ase.calculators.calculator import names as calc_names, get_calculator_class
from ase.cli.info import print_info
from ase.cli.main import CLIError

# asap is special, being the only calculator that may not be installed.
# But we want that for performance in some tests.
always_enabled_calculators = set(
    ['asap', 'eam', 'emt', 'ff', 'lj', 'morse', 'tip3p', 'tip4p']
)

# XXX test_calculator_names is edited from conftest
test_calculator_names = list(always_enabled_calculators)
testdir = Path(__file__).parent
datadir = (testdir / 'data').resolve()


def all_test_modules_and_groups():
    names = []
    groups = {}
    for abspath in testdir.rglob('test_*.py'):
        path = abspath.relative_to(testdir)
        name = str(path).rsplit('.', 1)[0].replace('/', '.')
        if str(path.parent) != '.':
            groupname = str(path.parent).replace('/', '.')
            groups.setdefault(groupname, []).append(name)
        else:
            names.append(name)
    return names, groups


def disable_calculators(names):
    import pytest
    for name in names:
        if name in always_enabled_calculators:
            continue
        try:
            cls = get_calculator_class(name)
        except ImportError:
            pass
        else:
            def get_mock_init(name):
                def mock_init(obj, *args, **kwargs):
                    pytest.skip(f'use --calculators={name} to enable')
                return mock_init

            def mock_del(obj):
                pass
            cls.__init__ = get_mock_init(name)
            cls.__del__ = mock_del


class must_raise:
    """Context manager for checking raising of exceptions."""
    def __init__(self, exception):
        self.exception = exception

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, tb):
        if exc_type is None:
            raise RuntimeError('Failed to fail: ' + str(self.exception))
        return issubclass(exc_type, self.exception)


def test(calculators=tuple(), jobs=0, verbose=False,
         stream='ignored', strict='ignored'):
    """Run the tests programmatically.

    This is here for compatibility and perhaps convenience."""
    from ase.cli.main import main

    if stream != 'ignored':
        warnings.warn('Ignoring old "stream" keyword', FutureWarning)
    if strict != 'ignored':
        warnings.warn('Ignoring old "strict" keyword', FutureWarning)

    args = ['test']
    if verbose:
        args += ['--verbose']
    if calculators:
        args += ['--calculators={}'.format(','.join(calculators))]
    if jobs:
        args += '--jobs={}'.format(jobs)

    main(args=args)


def have_module(module):
    return importlib.find_loader(module) is not None


MULTIPROCESSING_MAX_WORKERS = 32
MULTIPROCESSING_DISABLED = 0
MULTIPROCESSING_AUTO = -1


def choose_how_many_workers(jobs):
    if jobs == MULTIPROCESSING_AUTO:
        if have_module('xdist'):
            jobs = min(cpu_count(), MULTIPROCESSING_MAX_WORKERS)
        else:
            jobs = MULTIPROCESSING_DISABLED
    return jobs


class CLICommand:
    """Run ASE's test-suite.

    Requires the pytest package.  pytest-xdist is recommended
    in addition as the tests will then run in parallel.
    """

    @staticmethod
    def add_arguments(parser):
        parser.add_argument(
            '-c', '--calculators', default='',
            help='comma-separated list of calculators to test')
        parser.add_argument('--list', action='store_true',
                            help='print all tests and exit')
        parser.add_argument('--list-calculators', action='store_true',
                            help='print all calculator names and exit')
        parser.add_argument(
            '-j', '--jobs', type=int, metavar='N',
            default=MULTIPROCESSING_AUTO,
            help='number of worker processes.  If pytest-xdist is available,'
            ' defaults to all available processors up to a maximum of {}.  '
            '0 disables multiprocessing'
            .format(MULTIPROCESSING_MAX_WORKERS))
        parser.add_argument('-v', '--verbose', action='store_true',
                            help='write test outputs to stdout.  '
                            'Mostly useful when inspecting a single test')
        parser.add_argument('--strict', action='store_true',
                            help='convert warnings to errors.  '
                            'This option currently has no effect')
        parser.add_argument('--fast', action='store_true',
                            help='skip slow tests')
        parser.add_argument('--coverage', action='store_true',
                            help='measure code coverage.  '
                            'Requires pytest-cov')
        parser.add_argument('--nogui', action='store_true',
                            help='do not run graphical tests')
        parser.add_argument('tests', nargs='*',
                            help='specify particular test files '
                            'or directories')
        parser.add_argument('--pytest', nargs=argparse.REMAINDER,
                            help='forward all remaining arguments to pytest.  '
                            'See pytest --help')

    @staticmethod
    def run(args):
        print_info()
        print()

        if args.list_calculators:
            for name in calc_names:
                print(name)
            sys.exit(0)

        if args.nogui:
            os.environ.pop('DISPLAY')

        pytest_args = ['--pyargs', '-v']

        def add_args(*args):
            pytest_args.extend(args)

        if args.list:
            add_args('--collect-only')

        jobs = choose_how_many_workers(args.jobs)
        if jobs:
            add_args('--numprocesses={}'.format(jobs))

        if args.fast:
            add_args('-m', 'not slow')

        if args.coverage:
            add_args('--cov=ase',
                     '--cov-config=.coveragerc',
                     '--cov-report=term',
                     '--cov-report=html')

        if args.tests:
            names, groups = all_test_modules_and_groups()

            testnames = []
            for arg in args.tests:
                if arg in groups:
                    testnames += groups[arg]
                else:
                    testnames.append(arg)

            for testname in testnames:
                add_args('ase.test.{}'.format(testname))
        else:
            add_args('ase.test')

        if args.calculators:
            add_args(f'--calculators={args.calculators}')

        if args.verbose:
            add_args('--capture=no')

        if args.pytest:
            add_args(*args.pytest)

        print()
        calcstring = args.calculators if args.calculators else 'none'
        print('Enabled calculators: {}'.format(calcstring))
        print()
        print('About to run pytest with these parameters:')
        for line in pytest_args:
            print('    ' + line)

        if not have_module('pytest'):
            raise CLIError('Cannot import pytest; please install pytest '
                           'to run tests')

        # We run pytest through Popen rather than pytest.main().
        #
        # This is because some ASE modules were already imported and
        # would interfere with code coverage measurement.
        # (Flush so we don't get our stream mixed with the pytest output)
        sys.stdout.flush()
        proc = Popen([sys.executable, '-m', 'pytest'] + pytest_args,
                     cwd=str(testdir))
        exitcode = proc.wait()
        sys.exit(exitcode)
