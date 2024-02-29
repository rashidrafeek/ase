import re
from collections import Counter
from itertools import product, chain, combinations
from typing import Union

from fractions import Fraction
import numpy as np
import matplotlib.pyplot as plt

from ase.units import kB
from ase.formula import Formula


CONST = kB * np.log(10)  # Nernst constant

PREDEF_ENERGIES = {      # Default chemical potentials
    'H+': 0.0,           # for water, protons and electrons
    'e-': 0.0,
    'H2O': -2.4583
}

U_STD_AGCL = 0.222  # Standard redox potential of AgCl electrode
U_STD_SCE = 0.244   # Standard redox potential of SCE electrode


def initialize_refs(refs_dct):
    """Convert dictionary entries to Species instances"""
    refs = {}
    for name, energy in refs_dct.items():
        spec = Species(name)
        spec.set_chemical_potential(energy, None)
        refs[spec.name] = spec
    return refs


def get_product_combos(reactant, refs):
    """Obtain, from the available references, different combinations
       of products that are compatible with valid (electro)chemical reactions.
    """
    array = [[] for i in range(len(reactant.elements))]
    for ref in refs.values():
        contained = ref.contains(reactant.elements)
        for w in np.argwhere(contained).flatten():
            array[w].append(ref)
    return [np.unique(combo) for combo in product(*array)]


def get_phases(reactant, refs, T, conc, counter, tol=1e-3):
    """Obtain all the possible decomposition pathways
       for a given reactant as a collection of RedOx objects.
    """
    from scipy.linalg import null_space

    phases = []
    phase_matrix = []
    reac_elem = [-reactant._count_array(reactant.elements)]

    for products in get_product_combos(reactant, refs):
        prod_elem = [p._count_array(reactant.elements) for p in products]
        elem_matrix = np.array(reac_elem + prod_elem).T
        coeffs = null_space(elem_matrix).flatten()
        if len(coeffs) > 0 and all(coeffs > tol):
            coeffs /= coeffs[0]
            coeffs[0] = -coeffs[0]
            species = (reactant, *products)
            phase = RedOx(species, coeffs, T, conc, counter)
            phases.append(phase)
            phase_matrix.append(phase._vector)

    if len(phase_matrix) == 0:
        raise ValueError(
            "No valid decomposition pathways have been found" +
            " given this set of references."
        )

    return phases, np.array(phase_matrix).astype('float64')


def get_main_products(species):
    """Obtain the reaction products excluded protons,
       water and electrons.
    """
    return [spec for spec, coef in species.items()
            if coef > 0 and spec not in ['H+', 'H2O', 'e-']]


def format_label(species):
    """Obtain phase labels formatted in LaTeX math style."""
    formatted = []
    for prod in get_main_products(species):
        label = re.sub(r'(\S)([+-]+)', r'\1$^{\2}$', prod)
        label = re.sub(r'(\d+)', r'$_{\1}$', label)
        for symbol in ['+', '-']:
            count = label.count(symbol)
            if count > 1:
                label = label.replace(count * symbol, f'{count}{symbol}')
            if count == 1:
                label = label.replace(count * symbol, symbol)
        formatted.append(label)
    label = ', '.join(f for f in formatted)
    return label


def make_coeff_nice(coeff, max_denom):
    """Convert a fraction into a string while limiting the denominator"""
    frac = abs(Fraction(coeff).limit_denominator(max_denom))
    if frac.numerator == frac.denominator:
        return ''
    return str(frac)


def add_numbers(ax, text):
    """Add number identifiers to the different domains of a Pourbaix diagram."""
    import matplotlib.patheffects as pfx
    for i, (x, y, _) in enumerate(text):
        txt = ax.text(
            y, x, f'{i}',
            fontsize=20,
            horizontalalignment='center'
        )
        txt.set_path_effects([pfx.withStroke(linewidth=2.0, foreground='w')])
    return


def add_labels(ax, text):
    """Add phase labels to the different domains of a Pourbaix diagram."""
    import matplotlib.patheffects as pfx
    for i, (x, y, species) in enumerate(text):
        label = format_label(species)
        annotation = ax.annotate(
            label, xy=(y, x), color='w',
            fontsize=16, horizontalalignment='center'
        )
        annotation.set_path_effects(
            [pfx.withStroke(linewidth=2.0, foreground='k')]
        )
        annotation.draggable()
        ax.add_artist(annotation)
    return


def add_text(text, offset=0.0):
    """Add phase labels to the right of the diagram"""
    import textwrap

    textlines = []
    for i, (_, _, species) in enumerate(text):
        label = format_label(species)
        textlines.append(
            textwrap.fill(
                f'({i})  {label}',
                width=40,
                subsequent_indent='      '
            )
        )
    text = "\n".join(textlines)
    plt.gcf().text(
        0.75 + offset, 0.5,
        text,
        fontsize=16,
        va='center',
        ha='left')
    return 0


def add_redox_lines(axes, pH, counter, color='k'):
    """Add water redox potentials to a Pourbaix diagram"""
    const = - 0.5 * PREDEF_ENERGIES['H2O']
    corr = {
        'SHE': 0,
        'RHE': 0,
        'Pt': 0,
        'AgCl': -U_STD_AGCL,
        'SCE': -U_STD_SCE,
    }
    kwargs = {
        'c': color,
        'ls': '--',
        'zorder': 2
    }
    if counter in ['SHE', 'AgCl', 'SCE']:
        slope = -59.2e-3
        axes.plot(pH, slope * pH + corr[counter], **kwargs)
        axes.plot(pH, slope * pH + const + corr[counter], **kwargs)
    elif counter in ['Pt', 'RHE']:
        axes.axhline(0 + corr[counter], **kwargs)
        axes.axhline(const + corr[counter], **kwargs)
    else:
        raise ValueError('The specified counter electrode doesnt exist')
    return 0


class Species:
    """Class representing an individual chemical species,
       grouping relevant properties.

    Initialization
    --------------

    formula: str
        The chemical formula of the species (e.g. ``ZnO``).
        For solid species, the formula is automatically reduced to the
        unit formula, and the chemical potential normalized accordingly.
        Acqueous species are specified by expliciting
        all the positive or negative charges and by appending ``(aq)``.
        Parentheses for grouping functional groups are acceptable.
            e.g.
                Be3(OH)3[3+]    ➜  wrong
                Be3(OH)3+++     ➜  wrong
                Be3(OH)3+++(aq) ➜  correct
                Be3O3H3+++(aq)  ➜  correct
    fmt: str
        Formula formatting according to the available options in
        ase.formula.Formula

    """
    def __init__(self, formula, fmt='metal'):
        self.aq = formula.endswith('(aq)')
        formula_strip = formula.replace('(aq)', '').rstrip('+-')
        self.charge = formula.count('+') - formula.count('-')
        formula_obj = Formula(formula_strip, format=fmt)
        self._count = formula_obj.count()

        if self.aq:
            self.name = formula
            self.n_fu = 1
            self.count = self._count
        else:
            reduced, self.n_fu = formula_obj.reduce()
            self.count = reduced.count()
            self.name = str(reduced)

        self._elements = [elem for elem in self.count]
        self.elements = [
            elem for elem in self._elements if elem not in ['H', 'O']
        ]
        self.natoms = sum(self.count.values())
        self.energy = None
        self.mu = None

    def get_chemsys(self):
        """Get the possible combinations of elements based on the stoichiometry.
           Useful for database queries.
        """
        elements = set(self.count.keys())
        elements.update(['H', 'O'])
        chemsys = list(
            chain.from_iterable(
                [combinations(elements, i+1) for i,_ in enumerate(list(elements))]
            )
        )
        return chemsys

    def balance_electrochemistry(self):
        """Obtain number of H2O, H+, e- "carried" by the species
           in electrochemical reactions.
        """
        n_H2O = -self.count.get('O', 0)
        n_H = -2 * n_H2O - self.count.get('H', 0)
        n_e = n_H + self.charge
        return n_H2O, n_H, n_e

    def _count_array(self, elements):
        return np.array([self.count.get(e, 0) for e in elements])

    def contains(self, elements):
        return [True if elem in self.elements else False for elem in elements]

    def get_fractional_composition(self, elements):
        """Obtain the fractional content of each element."""
        N_all = sum(self.count.values())
        N_elem = sum([self.count.get(e, 0) for e in elements])
        return N_elem / N_all

    def get_formation_energy(self, energy, refs):
        """Evaluate the formation energy.

        Requires the material chemical potential (e.g. DFT total energy)
        and a dictionary with the elemental references and corresponding
        chemical potentials.
        """
        elem_energy = sum([refs[s] * n for s, n in self._count.items()])
        hof = (energy - elem_energy) / self.n_fu
        return hof

    def set_chemical_potential(self, energy, refs=None):
        """Set the chemical potential of the species.

        If a reference dictionary with the elements and their chemical potentials
        is provided, the chemical potential will be calculated by as a formation
        energy. Otherwise the provided energy will be used directly as the
        chemical potential.
        """
        self.energy = energy
        if refs is None:
            self.mu = energy / self.n_fu
        else:
            self.mu = self.get_formation_energy(energy, refs)

    def __repr__(self):
        return f'({self.name}, μ={self.mu})'

    def __lt__(self, other):
        return self.name < other.name

    def __gt__(self, other):
        return self.name > other.name


class RedOx:
    def __init__(self, species, coeffs,
                 T=298.15, conc=1e-6,
                 counter='SHE'):
        """RedOx class representing an (electro)chemical reaction.

        Initialization
        --------------

        species
            The reactant and products excluded H2O, protons and electrons.

        coeffs
            The stoichiometric coefficients of the above species.
            Positive coefficients are associated with the products,
            negative coefficients with the reactants.

        T
            The temperature in Kelvin.

        conc
            The concentration of ionic species.

        counter: str
            The counter electrode. Default: SHE.
            available options: SHE, RHE, AgCl, Pt.


        Relevant methods
        ----------------

        equation()
            Print the chemical equation of the reaction.

        get_free_energy(U, pH):
            Obtain the reaction free energy at a given applied potential U and pH

        """

        alpha = CONST * T   # 0.059 eV @ T=298.15K
        const_term = 0
        pH_term = 0
        U_term = 0
        self.species = Counter()

        for spec, coef in zip(species, coeffs):
            self.species[spec.name] = coef
            amounts = spec.balance_electrochemistry()

            const_term += coef * ( \
                spec.mu + alpha * (spec.aq * np.log10(conc)))
            pH_term += - coef * alpha * amounts[1]
            U_term += - coef * amounts[2]

            for name, n in zip(['H2O', 'H+', 'e-'], amounts):
                const_term += coef * n * PREDEF_ENERGIES[name]
                self.species[name] += coef * n

        const_corr, pH_corr = self.get_counter_correction(counter, alpha)
        self._vector = [
            float(const_term + const_corr),
            float(U_term),
            float(pH_term + pH_corr)
        ]

    def get_counter_correction(self, counter, alpha):
        """Correct the constant and pH contributions to the reaction free energy
           based on the counter electrode of choice and the temperature
           (alpha=k_B*T*ln(10))
        """
        n_e = self.species['e-']
        gibbs_corr = 0.0
        pH_corr = 0.0
        if counter in ['RHE', 'Pt']:
            pH_corr += n_e * alpha
            if counter == 'Pt' and n_e < 0:
                gibbs_corr +=  n_e * 0.5 * PREDEF_ENERGIES['H2O']
        if counter == 'AgCl':
            gibbs_corr -= n_e * U_STD_AGCL
        if counter == 'SCE':
            gibbs_corr -= n_e * U_STD_SCE

        return gibbs_corr, pH_corr

    def equation(self, max_denom=50):
        """Print the chemical reaction."""

        reactants = []
        products = []
        for s, n in self.species.items():
            if abs(n) <= 1e-6:
                continue
            substr = f'{make_coeff_nice(n, max_denom)}{s}'
            if n > 0:
                products.append(substr)
            else:
                reactants.append(substr)

        return "  ➜  ".join([" + ".join(reactants), " + ".join(products)])

    def get_free_energy(self, U, pH):
        """Evaluate the reaction free energy at a given applied potential U and pH"""
        return self._vector[0] + self._vector[1]*U + self._vector[2]*pH


class Pourbaix:
    """Pourbaix class for acqueous stability evaluations.

    Allows to determine the most stable phase in a given set
    of pH and potential conditions and to evaluate a complete diagram.

    Initialization
    --------------

    material_name: str
        The formula of the target material. It is preferrable
        to provide the reduced formula (e.g. RuO2 instad of Ru2O4).

    refs_dct: dict
        A dictionary containing the formulae of the target material
        and its competing phases (solid and/or ionic) as keys,
        and their formation energies as values.

    T: float
        Temperature in Kelvin. Default: 298.15 K.

    conc: float
        Concentration of the ionic species. Default: 1e-6 mol/L.

    counter: str
        The counter electrode. Default: SHE.
        available options: SHE, RHE, AgCl, Pt.


    Relevant methods
    ----------------

    get_pourbaix_energy(U, pH)
        obtain the energy of the target material
        relative to the most stable phase at a given potential U and pH.
        If negative, the target material can be regarded as stable.
    plot(**kwargs)
        plot a complete Pourbaix diagram in a given pH and potential window.


    Relevant attributes
    -------------------

    material: Species
        the target material as a Species object

    phases: list[RedOx]
        the available decomposition reactions of the target material
        into its competing phases as a list of RedOx objects.

    """
    def __init__(self,
            material_name:str,
            refs_dct:dict,
            T:float=298.15,
            conc:float=1.0e-6,
            counter:str='SHE'
        ):

        refs = initialize_refs(refs_dct)

        try:
            self.material = refs.pop(material_name)
        except KeyError:
            self.material = refs.pop(Species(material_name).name)

        self.counter = counter
        self.phases, phase_matrix = get_phases(
            self.material, refs, T, conc, counter
        )
        self._const = phase_matrix[:, 0]
        self._var = phase_matrix[:, 1:]

    def _decompose(self, U, pH):
        """Evaluate the reaction energy for decomposing
           the target material into each of the available products
           at a given pH and applied potential.
        """
        return self._const + np.dot(self._var, [U, pH])

    def _get_pourbaix_energy(self, U, pH):
        """Evaluate the Pourbaix energy"""
        energies = self._decompose(U, pH)
        i_min = np.argmin(energies)
        return -energies[i_min], i_min

    def get_pourbaix_energy(self, U, pH, verbose=True):
        """Evaluate the Pourbaix energy, print info about
        the most stable phase and the corresponding energy at
        a given potential U and pH.

        The Pourbaix energy represents the energy of the target material
        relative to the most stable competing phase. If negative,
        the target material can be considered as stable.
        """
        energy, index = self._get_pourbaix_energy(U, pH)
        phase = self.phases[index]
        if verbose:
            if energy <= 0.0:
                print(f'{self.material.name} is stable.')
            else:
                print(f'Stable phase: \n{phase.equation()}')
            print(f'Energy: {energy:.3f} eV')
        return energy, phase

    def get_equations(self, contains: Union[str,None]=None):
        """Print the chemical reactions of the available phases.

        the argument `contains' allows to filter for phases containing a
        particular species
            e.g. get_equations(contains='ZnO')
        """
        for i, p in enumerate(self.phases):
            if contains and contains not in p.species:
                continue
            print(f'{i}) {p.equation()}')

    def get_diagrams(self, U, pH):
        """Actual evaluation of the complete diagram

        Returns
        -------

        pour:
            The stability domains of the diagram on the pH vs. U grid.
            domains are represented by indexes (as integers)
            that map to Pourbaix.phases
            the target material is stable (index=-1)

        meta:
            The Pourbaix energy on the pH vs. U grid.

        text:
            The coordinates and phases information for
            text placement on the diagram.

        """
        pour = np.zeros((len(U), len(pH)))
        meta = pour.copy()

        for i, u in enumerate(U):
            for j, p in enumerate(pH):
                meta[i, j], pour[i, j] = self._get_pourbaix_energy(u, p)

        # Identifying the region where the target material
        # is stable and updating the diagram accordingly
        where_stable = (meta <= 0)
        pour[where_stable] = -1

        text = []
        domains = [int(i) for i in np.unique(pour)]
        for phase_id in domains:
            if phase_id == -1:
                where = where_stable
                txt = {self.material.name: 1}
            else:
                where = (pour == phase_id)
                phase = self.phases[int(phase_id)]
                txt = phase.species
            x = np.dot(where.sum(1), U) / where.sum()
            y = np.dot(where.sum(0), pH) / where.sum()
            text.append((x, y, txt))

        return pour, meta, text, domains

    def get_phase_boundaries(self, phrange, urange, domains, tol=1e-6):
        """Plane intersection method for finding
           the boundaries between phases seen in the final plot.

        Returns a list of tuples, each representing a single boundary,
        of the form ([[x1, x2], [y1, y2]], [id1, id2]), namely the
        x and y coordinates of the simplices connected by the boundary
        and the id's of the phases at each side of the boundary.
        """
        from itertools import combinations
        from collections import defaultdict

        # Planes identifying the diagram frame
        planes = [(np.array([0.0, 1.0, 0.0]), min(urange),  'b'),  # Bottom
                  (np.array([0.0, 1.0, 0.0]), max(urange),  't'),  # Top
                  (np.array([1.0, 0.0, 0.0]), min(phrange), 'l'),  # Left
                  (np.array([1.0, 0.0, 0.0]), max(phrange), 'r')]  # Right

        # Planes associated with the stable domains of the diagram.
        # Given x=pH, y=U, z=E_pbx=-DeltaG, each plane has expression:
        # _vector[2]*x + _vector[1]*y + z = -_vector[0]
        # The region where the target material is stable
        # (id=-1, if present) is delimited by the xy plane (Epbx=0)
        for d in domains:
            if d == -1:
                plane = np.array([0.0, 0.0, 1.0])
                const = 0.0
            else:
                pvec = self.phases[d]._vector
                plane = np.array([pvec[2], pvec[1], 1])
                const = -pvec[0]
            planes.append((plane, const, d))

        # The simplices are found from the intersection points between
        # all possible plane triplets. If the z coordinate of the point
        # matches the corresponding pourbaix energy,
        # then the point is a simplex.
        simplices = []
        for (p1, c1, id1), (p2, c2, id2), (p3, c3, id3) in \
                combinations(planes, 3):
            A = np.vstack((p1, p2, p3))
            c = np.array([c1, c2, c3])
            ids = (id1, id2, id3)
            try:
                pt = np.dot(np.linalg.inv(A), c)
                Epbx = self._get_pourbaix_energy(pt[1], pt[0])[0]
                if pt[2] >= -tol and \
                   np.isclose(Epbx, pt[2], rtol=0, atol=tol) and \
                   min(phrange)-tol <= pt[0] <= max(phrange)+tol and \
                   min(urange)-tol  <= pt[1] <= max(urange)+tol:
                        simplex = np.round(pt[:2], 3)
                        simplices.append((simplex, ids))

            # triplets containing parallel planes raise a LinAlgError
            # and are automatically excluded.
            except np.linalg.LinAlgError:
                continue

        # The final segments to plot on the diagram are found from
        # the pairs of unique simplices that have two neighboring phases
        # in common, diagram frame excluded.
        duplicate_filter = defaultdict(int)
        for (s1, id1), (s2, id2) in combinations(simplices, 2):

            # common neighboring phases
            common = list(set(id1).intersection(id2))

            # Simplices have to be distinct
            cond1 = not np.allclose(s1, s2, rtol=0, atol=tol)
            # Only two phases in common...
            cond2 = (len(common) == 2)
            # ...Diagram frame excluded
            cond3 = not any(np.isin(common, ['b', 't', 'l', 'r']))

            # Filtering out duplicates
            if all((cond1, cond2, cond3)):
                testarray = sorted(
                    [s1, s2], key=lambda s: (s[0], s[1])
                )
                testarray.append(sorted(common))
                testarray = np.array(testarray).flatten()
                duplicate_filter[tuple(testarray)] += 1

        segments = []
        for segment in duplicate_filter:
            coords, phases = np.split(np.array(segment), [4])
            segments.append((
                coords.reshape(2, 2).T,
                list(phases.astype(int))
            ))
        return segments


    def _draw_diagram_axes(
            self,
            Urange, pHrange,
            npoints, cap,
            figsize, normalize,
            include_text,
            include_h2o,
            labeltype, cmap):
        """Backend for drawing Pourbaix diagrams"""

        pH = np.linspace(*pHrange, num=npoints)
        U = np.linspace(*Urange, num=npoints)

        pour, meta, text, domains = self.get_diagrams(U, pH)

        if normalize:
            meta /= self.material.natoms

        ax = plt.figure(figsize=figsize).add_subplot(111)
        extent = [*pHrange, *Urange]

        plt.subplots_adjust(
            left=0.1, right=0.97,
            top=0.97, bottom=0.14
        )

        if isinstance(cap, list):
            vmin = cap[0]
            vmax = cap[1]
        else:
            vmin = -cap
            vmax = cap

        colorplot = ax.imshow(
            meta, cmap=cmap,
            extent=extent,
            vmin=vmin, vmax=vmax,
            origin='lower', aspect='auto',
            interpolation='gaussian'
        )

        cbar = plt.gcf().colorbar(
               colorplot,
               ax=ax,
               pad=0.02
        )

        bounds = self.get_phase_boundaries(
            pHrange, Urange, domains
        )
        for coords,_ in bounds:
            ax.plot(coords[0], coords[1], '-', c='k', lw=1.0)

        if labeltype == 'numbers':
            add_numbers(ax, text)
        elif labeltype == 'phases':
            add_labels(ax, text)
        elif labeltype == None:
            pass
        else:
            raise ValueError("The provided label type doesn't exist")

        if include_h2o:
            add_redox_lines(ax, pH, self.counter, 'w')

        ax.set_xlim(*pHrange)
        ax.set_ylim(*Urange)
        ax.set_xlabel('pH', fontsize=22)
        ax.set_ylabel(r'$\it{U}$' + f' vs. {self.counter} (V)', fontsize=22)
        ax.set_xticks(np.arange(pHrange[0], pHrange[1] + 1, 2))
        ax.set_yticks(np.arange(Urange[0], Urange[1] + 1, 1))
        ax.xaxis.set_tick_params(width=1.5, length=5)
        ax.yaxis.set_tick_params(width=1.5, length=5)
        plt.xticks(fontsize=22)
        plt.yticks(fontsize=22)

        ticks = np.linspace(vmin, vmax, num=5)
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(ticks)
        cbar.outline.set_linewidth(1.5)
        cbar.ax.tick_params(labelsize=20, width=1.5, length=5)
        cbar.ax.set_ylabel(r'$E_{pbx}$ (eV/atom)', fontsize=20)

        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(1.5)

        if include_text:
            plt.subplots_adjust(right=0.75)
            add_text(text, offset=0.05)
            return ax, cbar

        plt.tight_layout()
        return ax, cbar

    def plot(self,
             Urange=[-2, 2],
             pHrange=[0, 14],
             npoints=300,
             cap=1.0,
             figsize=[12, 6],
             normalize=True,
             include_text=True,
             include_h2o=False,
             labeltype='numbers',
             cmap="RdYlGn_r",
             savefig=None,
             show=True):
        """Plot a complete Pourbaix diagram.

        Keyword arguments
        -----------------

        Urange: list
            The potential range onto which to draw the diagram.

        pHrange: list
            The pH range onto which to draw the diagram.

        npoints: int
            The resolution of the diagram. Higher values
            mean higher resolution and thus higher compute times.

        cap: float/list
            If float, the limit (in both the positive and negative direction)
            of the Pourbaix energy colormap.
            If list, the first and second value determine the colormap limits.

        figsize: list
            The horizontal and vertical size of the graph.

        normalize: bool
            Normalize energies by the number of
            atoms in the target material unit formula.

        include_text: bool
            Report to the right of the diagram the main products
            associated with the stability domains.

        include_h20: bool
            Include in the diagram the stability domain of water.

        labeltype: str/None
            The labeling style of the diagram domains. Options:
                'numbers': just add numbers associated with the different phases,
                           the latter shown on the right if include_text=True.
                'phases':  Write the main products directly on the diagram.
                           These labels can be dragged around if the placement
                           is unsatisfactory. Redundant if include_text=True.
                 None:     Don't draw any labels.

        savefig: str/None
            If passed as a string, the figure will be saved with that name.

        show: bool
            Spawn a window showing the diagram.

        """
        ax, colorbar = self._draw_diagram_axes(
             Urange, pHrange,
             npoints, cap,
             figsize, normalize,
             include_text,
             include_h2o,
             labeltype, cmap)

        if savefig:
            plt.savefig(savefig)
        if show:
            plt.show()
