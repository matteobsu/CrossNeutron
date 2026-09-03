"""BeamMatter neutron cross-section calculations.

Pure-Python port of the numerical model in libnxs/nxs.c.

Runtime dependencies: numpy only.

The only crystallographic preprocessing expected from the material layer is
that every atom record provides its symmetry-expanded fractional positions.
This lets the calculation sum every hkl in the search cube directly instead
of reproducing SGInfo's systematic-absence/equivalence bookkeeping at runtime.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Iterable

import numpy as np

# Constants copied from libnxs/nxs.c
AVOGADRO = 6.0221417930e23
ATOMIC_MASS_U_KG = 1.6605402e-27
MASS_NEUTRON_KG = 1.6749e-27
ENERGY_MEV_A2 = 81.8042531017
DEFAULT_MAX_HKL = 20


@dataclass(frozen=True, slots=True)
class _PreparedAtom:
    label: str
    b_coherent_fm: float
    sigma_incoherent_barn: float
    sigma_absorption_2200_barn: float
    molar_mass_g_mol: float
    debye_temperature_k: float
    positions: tuple[tuple[float, float, float], ...]
    phi_1: float
    phi_3: float
    b_iso: float

    @property
    def multiplicity(self) -> int:
        return len(self.positions)


@dataclass(frozen=True, slots=True)
class _PreparedMaterial:
    material_id: str
    name: str
    formula: str | None
    space_group: str
    crystal_system: int
    volume_a3: float
    mass_g_mol_unit_cell: float
    density_g_cm3: float
    n_atoms_unit_cell: int
    avg_sigma_coherent_barn: float
    avg_sigma_incoherent_barn: float
    atoms: tuple[_PreparedAtom, ...]
    reflection_d_a: np.ndarray
    reflection_f_squared_fm2: np.ndarray


def calculate_cross_sections(
    *,
    materials: list[Any],
    wavelength_min_a: float,
    wavelength_max_a: float,
    points: int,
    temperature_k: float,
    components: list[str],
    max_hkl: int | None = None,
) -> dict[str, Any]:
    """Main API-facing calculation function."""
    wavelength_min_a = float(wavelength_min_a)
    wavelength_max_a = float(wavelength_max_a)
    temperature_k = float(temperature_k)
    points = int(points)
    max_hkl = DEFAULT_MAX_HKL if max_hkl is None else int(max_hkl)

    if wavelength_min_a <= 0.0:
        raise ValueError("wavelength_min_a must be greater than zero.")
    if wavelength_max_a <= wavelength_min_a:
        raise ValueError("wavelength_max_a must be greater than wavelength_min_a.")
    if points < 2:
        raise ValueError("points must be at least 2.")
    if temperature_k <= 0.0:
        raise ValueError("temperature_k must be greater than zero.")
    if max_hkl < 1:
        raise ValueError("max_hkl must be greater than zero.")
    if not materials:
        raise ValueError("At least one material is required.")

    allowed = {
        "total",
        "coherent_elastic",
        "incoherent_elastic",
        "coherent_inelastic",
        "incoherent_inelastic",
        "total_inelastic",
        "absorption",
    }
    requested = []
    for component in components:
        key = str(component).strip().lower()
        if key not in allowed:
            raise ValueError(f"Unknown cross-section component '{component}'.")
        if key not in requested:
            requested.append(key)
    if not requested:
        raise ValueError("At least one cross-section component is required.")

    wavelength = np.linspace(
        wavelength_min_a,
        wavelength_max_a,
        points,
        dtype=np.float64,
    )
    energy = wavelength_to_energy_mev(wavelength)

    series: list[dict[str, Any]] = []
    material_summaries: list[dict[str, Any]] = []

    for material in materials:
        prepared = _prepare_material(material, temperature_k, max_hkl)
        curves = _calculate_prepared_material(prepared, wavelength)

        for component in requested:
            series.append({
                "material_id": prepared.material_id,
                "material_name": prepared.name,
                "component": component,
                "name": f"{prepared.name} - {_component_label(component)}",
                "values_barn": curves[component].tolist(),
            })

        density_override = getattr(material, "density_override_g_cm3", None)
        material_summaries.append({
            "id": prepared.material_id,
            "name": prepared.name,
            "formula": prepared.formula,
            "space_group": prepared.space_group,
            "temperature_k": temperature_k,
            "max_hkl": max_hkl,
            "n_atoms_unit_cell": prepared.n_atoms_unit_cell,
            "volume_a3": prepared.volume_a3,
            "mass_g_mol_unit_cell": prepared.mass_g_mol_unit_cell,
            "calculated_density_g_cm3": prepared.density_g_cm3,
            "density_g_cm3": (
                float(density_override)
                if density_override is not None
                else prepared.density_g_cm3
            ),
            "avg_sigma_coherent_barn": prepared.avg_sigma_coherent_barn,
            "avg_sigma_incoherent_barn": prepared.avg_sigma_incoherent_barn,
        })

    return {
        "engine": {
            "name": "libnxs Python port",
            "reference_version": "1.2",
            "max_hkl": max_hkl,
        },
        "axes": {
            "wavelength_a": wavelength.tolist(),
            "energy_mev": energy.tolist(),
        },
        "series": series,
        "materials": material_summaries,
    }


def wavelength_to_energy_mev(wavelength_a: np.ndarray | float) -> np.ndarray:
    wavelength = np.asarray(wavelength_a, dtype=np.float64)
    if np.any(wavelength <= 0.0):
        raise ValueError("Wavelength must be greater than zero.")
    return ENERGY_MEV_A2 / np.square(wavelength)


def calc_phi_1(theta: float) -> float:
    """Direct port of calcPhi_1() in nxs.c."""
    theta = float(theta)
    if theta <= 0.0:
        raise ValueError("theta must be greater than zero.")

    step1 = 1.0
    step2 = 0.0
    n = 1.0
    a_n = 0.0

    while abs(step1 - step2) > 1.0e-6:
        step1 = step2
        exponent = n / theta
        step2 = 0.0 if exponent > 700.0 else 1.0 / (math.exp(exponent) * n * n)
        a_n += step2
        n += 1.0

    riemann_zeta_2 = math.pi * math.pi / 6.0
    i_m = (
        theta * math.log(1.0 - math.exp(-1.0 / theta))
        + theta * theta * (riemann_zeta_2 - a_n)
    )
    return 0.5 + 2.0 * i_m


def calc_phi_3(theta: float) -> float:
    """Direct port of calcPhi_3() in nxs.c."""
    theta = float(theta)
    if theta <= 0.0:
        raise ValueError("theta must be greater than zero.")

    step1 = 1.0
    step2 = 0.0
    n = 1.0
    a_n = 0.0

    while abs(step1 - step2) > 1.0e-6:
        step1 = step2
        exponent = n / theta
        if exponent > 700.0:
            step2 = 0.0
        else:
            step2 = (
                1.0 / (math.exp(exponent) * n * n)
                * (0.5 + theta / n + theta * theta / (n * n))
            )
        a_n += step2
        n += 1.0

    riemann_zeta_4 = math.pi ** 4 / 90.0
    i_m = (
        theta * math.log(1.0 - math.exp(-1.0 / theta))
        + 6.0 * theta * theta * (riemann_zeta_4 * theta * theta - a_n)
    )
    return 0.25 + 2.0 * i_m


def _prepare_material(material: Any, temperature_k: float, max_hkl: int) -> _PreparedMaterial:
    crystal_system = _crystal_system(material.space_group)
    volume = _cell_volume(material, crystal_system)
    if volume <= 0.0 or not math.isfinite(volume):
        raise ValueError(f"Material '{material.id}' has an invalid unit-cell volume.")

    prepared_atoms: list[_PreparedAtom] = []
    n_atoms = 0
    mass = 0.0

    for atom in material.atoms:
        if atom.debye_temperature_k <= 0.0:
            raise ValueError(
                f"Material '{material.id}', atom '{atom.label}': "
                "Debye temperature must be greater than zero for libnxs calculations."
            )

        positions = _expanded_positions(atom)
        multiplicity = len(positions)
        if multiplicity < 1:
            raise ValueError(
                f"Material '{material.id}', atom '{atom.label}' has no expanded positions."
            )

        theta = temperature_k / atom.debye_temperature_k
        phi_1 = calc_phi_1(theta)
        phi_3 = calc_phi_3(theta)
        b_iso = (
            5.7451121e3
            * phi_1
            / atom.molar_mass_g_mol
            / atom.debye_temperature_k
        )

        prepared_atoms.append(_PreparedAtom(
            label=atom.label,
            b_coherent_fm=float(atom.b_coherent_fm),
            sigma_incoherent_barn=float(atom.sigma_incoherent_barn),
            sigma_absorption_2200_barn=float(atom.sigma_absorption_2200_barn),
            molar_mass_g_mol=float(atom.molar_mass_g_mol),
            debye_temperature_k=float(atom.debye_temperature_k),
            positions=positions,
            phi_1=phi_1,
            phi_3=phi_3,
            b_iso=b_iso,
        ))

        n_atoms += multiplicity
        mass += multiplicity * atom.molar_mass_g_mol

    if n_atoms <= 0:
        raise ValueError(f"Material '{material.id}' contains no atoms.")

    density = mass / volume / AVOGADRO * 1.0e24

    mean_b_sum = 0.0
    sigma_inc_sum = 0.0
    b_square_sum = 0.0

    for atom in prepared_atoms:
        multiplicity = atom.multiplicity
        mean_b_sum += atom.b_coherent_fm * multiplicity
        sigma_inc_sum += atom.sigma_incoherent_barn * multiplicity
        b_square_sum += atom.b_coherent_fm ** 2 * multiplicity

    mean_b = mean_b_sum / n_atoms
    mean_b_square = b_square_sum / n_atoms
    avg_sigma_incoherent = sigma_inc_sum / n_atoms
    avg_sigma_coherent = mean_b * mean_b

    avg_sigma_incoherent += (
        0.04 * math.pi * (mean_b_square - avg_sigma_coherent)
    )
    avg_sigma_coherent *= 0.04 * math.pi

    d_values, f_squared = _build_reflection_sum_data(
        material=material,
        atoms=tuple(prepared_atoms),
        crystal_system=crystal_system,
        max_hkl=max_hkl,
    )

    return _PreparedMaterial(
        material_id=material.id,
        name=material.name,
        formula=material.formula,
        space_group=material.space_group,
        crystal_system=crystal_system,
        volume_a3=volume,
        mass_g_mol_unit_cell=mass,
        density_g_cm3=density,
        n_atoms_unit_cell=n_atoms,
        avg_sigma_coherent_barn=avg_sigma_coherent,
        avg_sigma_incoherent_barn=avg_sigma_incoherent,
        atoms=tuple(prepared_atoms),
        reflection_d_a=d_values,
        reflection_f_squared_fm2=f_squared,
    )


def _calculate_prepared_material(
    material: _PreparedMaterial,
    wavelength: np.ndarray,
) -> dict[str, np.ndarray]:
    coherent_elastic = _coherent_elastic(wavelength, material)
    incoherent_elastic = _incoherent_elastic(wavelength, material)
    incoherent_inelastic = _inelastic(wavelength, material, coherent=False)
    coherent_inelastic = _inelastic(wavelength, material, coherent=True)
    total_inelastic = _total_inelastic(wavelength, material)
    absorption = _absorption(wavelength, material)

    total = (
        coherent_elastic
        + incoherent_elastic
        + total_inelastic
        + absorption
    )

    return {
        "total": total,
        "coherent_elastic": coherent_elastic,
        "incoherent_elastic": incoherent_elastic,
        "coherent_inelastic": coherent_inelastic,
        "incoherent_inelastic": incoherent_inelastic,
        "total_inelastic": total_inelastic,
        "absorption": absorption,
    }


def _coherent_elastic(wavelength: np.ndarray, material: _PreparedMaterial) -> np.ndarray:
    """Vectorized equivalent of nxs_CoherentElastic()."""
    d = material.reflection_d_a
    f2 = material.reflection_f_squared_fm2
    if d.size == 0:
        return np.zeros_like(wavelength)

    # Sum every hkl individually. This is equivalent to NXS's
    # inequivalent-reflection * multiplicity sum when the material JSON
    # contains the complete symmetry-expanded atomic positions.
    order = np.argsort(d)[::-1]
    d = d[order]
    weighted = f2[order] * d
    edges = 2.0 * d
    prefix = np.concatenate(([0.0], np.cumsum(weighted)))

    # NXS condition: lambda - 2*d < 1E-6
    thresholds = wavelength - 1.0e-6
    count = np.searchsorted(-edges, -thresholds, side="right")
    sums = prefix[count]

    return (
        sums
        * 1.0e-2
        * np.square(wavelength)
        / (2.0 * material.volume_a3)
    )


def _absorption(wavelength: np.ndarray, material: _PreparedMaterial) -> np.ndarray:
    """Direct equivalent of nxs_Absorption()."""
    sigma = sum(
        atom.sigma_absorption_2200_barn * atom.multiplicity
        for atom in material.atoms
    )
    return sigma / 1.798 * wavelength


def _incoherent_elastic(wavelength: np.ndarray, material: _PreparedMaterial) -> np.ndarray:
    """Direct equivalent of nxs_IncoherentElastic()."""
    total = np.zeros_like(wavelength)
    lambda_sq = np.square(wavelength)

    for atom in material.atoms:
        value = lambda_sq / (2.0 * atom.b_iso)
        s_el_inc = value * (1.0 - np.exp(-1.0 / value))
        total += s_el_inc * atom.multiplicity

    return total * material.avg_sigma_incoherent_barn


def _inelastic(
    wavelength: np.ndarray,
    material: _PreparedMaterial,
    *,
    coherent: bool,
) -> np.ndarray:
    """Shared body of nxs_CoherentInelastic / nxs_IncoherentInelastic."""
    total = np.zeros_like(wavelength)
    lambda_sq = np.square(wavelength)

    for atom in material.atoms:
        phi1_phi3 = atom.phi_1 * atom.phi_3
        value = lambda_sq / (2.0 * atom.b_iso)
        s_el_inc = value * (1.0 - np.exp(-1.0 / value))
        a_ratio = atom.molar_mass_g_mol * ATOMIC_MASS_U_KG / MASS_NEUTRON_KG
        s_total_inc = (
            (a_ratio / (a_ratio + 1.0)) ** 2
            * (1.0 + 9.0 * phi1_phi3 * value / (a_ratio * a_ratio))
        )
        total += (s_total_inc - s_el_inc) * atom.multiplicity

    sigma = (
        material.avg_sigma_coherent_barn
        if coherent
        else material.avg_sigma_incoherent_barn
    )
    return total * sigma


def _total_inelastic(wavelength: np.ndarray, material: _PreparedMaterial) -> np.ndarray:
    """Direct equivalent of nxs_TotalInelastic()."""
    total = np.zeros_like(wavelength)
    lambda_sq = np.square(wavelength)

    for atom in material.atoms:
        phi1_phi3 = atom.phi_1 * atom.phi_3
        value = lambda_sq / (2.0 * atom.b_iso)
        s_el_inc = value * (1.0 - np.exp(-1.0 / value))
        a_ratio = atom.molar_mass_g_mol * ATOMIC_MASS_U_KG / MASS_NEUTRON_KG
        s_total_inc = (
            (a_ratio / (a_ratio + 1.0)) ** 2
            * (1.0 + 9.0 * phi1_phi3 * value / (a_ratio * a_ratio))
        )
        total += (s_total_inc - s_el_inc) * atom.multiplicity

    return total * (
        material.avg_sigma_coherent_barn
        + material.avg_sigma_incoherent_barn
    )


def _build_reflection_sum_data(
    *,
    material: Any,
    atoms: tuple[_PreparedAtom, ...],
    crystal_system: int,
    max_hkl: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build all hkl terms needed by the coherent-elastic powder sum."""
    axis = np.arange(-max_hkl, max_hkl + 1, dtype=np.int32)
    h, k, l = np.meshgrid(axis, axis, axis, indexing="ij")
    h = h.ravel()
    k = k.ravel()
    l = l.ravel()

    nonzero = (h != 0) | (k != 0) | (l != 0)
    h = h[nonzero]
    k = k[nonzero]
    l = l[nonzero]

    d = _d_spacing_array(material, crystal_system, h, k, l)
    valid = np.isfinite(d) & (d > 0.0)
    h = h[valid]
    k = k[valid]
    l = l[valid]
    d = d[valid]

    real = np.zeros(d.shape, dtype=np.float64)
    imag = np.zeros(d.shape, dtype=np.float64)

    for atom in atoms:
        exp_b_iso = np.exp(-atom.b_iso / (4.0 * np.square(d)))
        cos_sum = np.zeros_like(d)
        sin_sum = np.zeros_like(d)

        for x, y, z in atom.positions:
            # nxs_calcFSquare() special-cases x+y+z ~= 0 only to avoid
            # evaluating sin/cos; mathematically this is identical.
            phase = 2.0 * math.pi * (x * h + y * k + z * l)
            cos_sum += np.cos(phase)
            sin_sum += np.sin(phase)

        real += cos_sum * exp_b_iso * atom.b_coherent_fm
        imag += sin_sum * exp_b_iso * atom.b_coherent_fm

    f_squared = np.square(real) + np.square(imag)

    # SGInfo removes systematic absences before nxs_calcFSquare(). With
    # complete symmetry-expanded positions those reflections have F^2=0.
    # Remove numerical round-off only.
    keep = f_squared > 1.0e-20
    return d[keep], f_squared[keep]


def _d_spacing_array(
    material: Any,
    crystal_system: int,
    h: np.ndarray,
    k: np.ndarray,
    l: np.ndarray,
) -> np.ndarray:
    """Vectorized direct port of nxs_calcDhkl(). Angles stay as raw NXS values."""
    a = float(material.a)
    b = float(material.b)
    c = float(material.c)
    alpha = float(material.alpha)
    beta = float(material.beta)
    gamma = float(material.gamma)

    hf = h.astype(np.float64)
    kf = k.astype(np.float64)
    lf = l.astype(np.float64)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        if crystal_system == 7:  # cubic
            return a / np.sqrt(hf * hf + kf * kf + lf * lf)

        if crystal_system == 6:  # hexagonal
            return a / np.sqrt(
                4.0 / 3.0 * (hf * hf + hf * kf + kf * kf)
                + (a * a / (c * c)) * lf * lf
            )

        if crystal_system == 5:  # trigonal
            return (
                math.sqrt(3.0) * a * c
                / np.sqrt(
                    4.0 * (hf * hf + kf * kf + hf * kf) * c * c
                    + 3.0 * lf * lf * a * a
                )
            )

        if crystal_system == 4:  # tetragonal
            return a / np.sqrt(
                hf * hf + kf * kf + a * a / (c * c) * lf * lf
            )

        if crystal_system == 3:  # orthorhombic
            return 1.0 / np.sqrt(
                hf * hf / (a * a)
                + kf * kf / (b * b)
                + lf * lf / (c * c)
            )

        if crystal_system == 2:  # monoclinic
            numerator = a * b * c * math.sqrt(1.0 - math.cos(beta) ** 2)
            denominator = np.sqrt(
                b * b * c * c * hf * hf
                + a * a * c * c * kf * kf * math.sin(beta) ** 2
                + a * a * b * b * lf * lf
                - 2.0 * a * b * b * c * hf * lf * math.cos(beta)
            )
            return numerator / denominator

        if crystal_system == 1:  # triclinic
            t1 = b * b * c * c * math.sin(alpha)
            t2 = a * a * c * c * math.sin(beta)
            t3 = a * a * b * b * math.sin(gamma)
            t4 = a * b * c * c * (math.cos(alpha) * math.cos(beta) - math.cos(gamma))
            t5 = a * a * b * c * (math.cos(beta) * math.cos(gamma) - math.cos(alpha))
            t6 = b * b * c * c * (math.cos(gamma) * math.cos(alpha) - math.cos(beta))
            volume = _cell_volume(material, crystal_system)
            inv_d_sq = (
                t1 * hf * hf
                + t2 * kf * kf
                + t3 * lf * lf
                + 2.0 * t4 * hf * kf
                + 2.0 * t5 * kf * lf
                + 2.0 * t6 * hf * lf
            ) / (volume * volume)
            return np.sqrt(1.0 / inv_d_sq)

    return np.full(h.shape, np.nan, dtype=np.float64)


def _cell_volume(material: Any, crystal_system: int) -> float:
    """Direct port of the volume switch in nxs_initUnitCell()."""
    a = float(material.a)
    b = float(material.b)
    c = float(material.c)
    alpha = float(material.alpha)
    beta = float(material.beta)
    gamma = float(material.gamma)

    if crystal_system == 7:
        return a * a * a
    if crystal_system in (5, 6):
        return 0.866025 * a * a * c
    if crystal_system == 4:
        return a * a * c
    if crystal_system == 3:
        return a * b * c
    if crystal_system == 2:
        return a * b * c * math.sin(beta)
    if crystal_system == 1:
        return a * b * c * math.sqrt(
            1.0
            - math.cos(alpha) ** 2
            - math.cos(beta) ** 2
            - math.cos(gamma) ** 2
            + 2.0 * math.cos(alpha) * math.cos(beta) * math.cos(gamma)
        )
    return 0.0


def _crystal_system(space_group: str | int) -> int:
    """Map standard space-group number to the same 1..7 crystal-system codes used by SGInfo."""
    text = str(space_group).strip()
    try:
        number = int(text)
    except ValueError:
        compact = text.replace(" ", "").upper()
        # Legacy Al example: Hall symbol '-F 4 2 3'.
        if compact == "-F423":
            return 7
        raise ValueError(
            f"Unsupported non-numeric space group '{space_group}'. "
            "Store the material's crystal_system explicitly if additional Hall symbols are needed."
        )

    if 1 <= number <= 2:
        return 1
    if 3 <= number <= 15:
        return 2
    if 16 <= number <= 74:
        return 3
    if 75 <= number <= 142:
        return 4
    if 143 <= number <= 167:
        return 5
    if 168 <= number <= 194:
        return 6
    if 195 <= number <= 230:
        return 7
    raise ValueError(f"Invalid space-group number '{number}'.")


def _expanded_positions(atom: Any) -> tuple[tuple[float, float, float], ...]:
    """Read symmetry-expanded positions from the material record."""
    positions = getattr(atom, "positions", None)
    if positions is None:
        raise ValueError(
            f"Atom '{atom.label}' does not contain symmetry-expanded positions. "
            "Update materials.py/materials.json before calculating cross sections."
        )

    result = tuple(
        (float(position[0]), float(position[1]), float(position[2]))
        for position in positions
    )
    if not result:
        raise ValueError(f"Atom '{atom.label}' has an empty positions list.")
    return result


def _component_label(component: str) -> str:
    return {
        "total": "Total",
        "coherent_elastic": "Coherent elastic",
        "incoherent_elastic": "Incoherent elastic",
        "coherent_inelastic": "Coherent inelastic",
        "incoherent_inelastic": "Incoherent inelastic",
        "total_inelastic": "Total inelastic",
        "absorption": "Absorption",
    }[component]
