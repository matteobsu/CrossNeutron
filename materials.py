"""
CrossNeutron / BeamMatter material definitions.

Loads material data from:

    data/materials.json

The JSON contains the raw NXS material inputs plus pre-expanded
fractional atomic positions.

This module performs only:
    - JSON loading
    - validation
    - conversion into immutable Python data structures
    - material catalog generation

It does NOT calculate neutron cross sections.
That belongs in calculate.py.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MATERIALS_FILE = DATA_DIR / "materials.json"


@dataclass(frozen=True, slots=True)
class AtomDefinition:
    label: str

    b_coherent_fm: float
    sigma_incoherent_barn: float
    sigma_absorption_2200_barn: float
    molar_mass_g_mol: float
    debye_temperature_k: float

    x: float
    y: float
    z: float

    positions: tuple[
        tuple[float, float, float],
        ...
    ]


@dataclass(frozen=True, slots=True)
class MaterialDefinition:
    id: str
    name: str
    formula: str | None
    space_group: str

    a: float
    b: float
    c: float

    alpha: float
    beta: float
    gamma: float

    atoms: tuple[AtomDefinition, ...]

    density_override_g_cm3: float | None = None


@lru_cache(maxsize=1)
def _load_database() -> dict[str, Any]:
    if not MATERIALS_FILE.is_file():
        raise FileNotFoundError(
            f"Material database not found: {MATERIALS_FILE}"
        )

    with MATERIALS_FILE.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(
            "materials.json must contain a JSON object."
        )

    materials = data.get("materials")

    if not isinstance(materials, list):
        raise ValueError(
            "materials.json must contain a 'materials' array."
        )

    return data


@lru_cache(maxsize=1)
def _material_index() -> dict[str, dict[str, Any]]:
    database = _load_database()

    result: dict[str, dict[str, Any]] = {}

    for index, record in enumerate(database["materials"]):
        if not isinstance(record, dict):
            raise ValueError(
                f"materials[{index}] must be an object."
            )

        material_id = record.get("id")

        if not isinstance(material_id, str):
            raise ValueError(
                f"materials[{index}].id must be a string."
            )

        material_id = material_id.strip()

        if not material_id:
            raise ValueError(
                f"materials[{index}].id cannot be empty."
            )

        if material_id in result:
            raise ValueError(
                f"Duplicate material ID '{material_id}'."
            )

        result[material_id] = record

    return result


def get_material_catalog() -> list[dict[str, Any]]:
    """
    Return only materials that are currently usable by calculate.py.
    """

    result: list[dict[str, Any]] = []

    for material_id, record in _material_index().items():
        try:
            material = _parse_material_record(
                material_id,
                record,
                None,
            )
        except ValueError:
            continue

        if any(
            atom.debye_temperature_k <= 0.0
            for atom in material.atoms
        ):
            continue

        result.append({
            "id": material.id,
            "name": material.name,
            "formula": material.formula,
        })

    result.sort(
        key=lambda item: item["name"].lower()
    )

    return result


def resolve_material(
    material_id: str,
    density_override: float | None = None,
) -> MaterialDefinition:
    if not isinstance(material_id, str):
        raise ValueError(
            "material_id must be a string."
        )

    material_id = material_id.strip()

    if not material_id:
        raise ValueError(
            "material_id cannot be empty."
        )

    try:
        record = _material_index()[material_id]
    except KeyError as exc:
        raise KeyError(
            f"Unknown material '{material_id}'."
        ) from exc

    return _parse_material_record(
        material_id,
        record,
        density_override,
    )


def get_material(
    material_id: str,
) -> MaterialDefinition:
    return resolve_material(material_id)


def clear_material_cache() -> None:
    _material_index.cache_clear()
    _load_database.cache_clear()


def _parse_material_record(
    material_id: str,
    record: dict[str, Any],
    density_override: float | None,
) -> MaterialDefinition:

    name = record.get("name", material_id)

    if not isinstance(name, str) or not name.strip():
        raise ValueError(
            f"Material '{material_id}' has an invalid name."
        )

    name = name.strip()

    formula = record.get("formula")

    if formula is not None:
        if not isinstance(formula, str):
            raise ValueError(
                f"Material '{material_id}' has an invalid formula."
            )

        formula = formula.strip() or None

    space_group = record.get("space_group")

    if not isinstance(space_group, (str, int)):
        raise ValueError(
            f"Material '{material_id}' has an invalid space_group."
        )

    space_group = str(space_group).strip()

    if not space_group:
        raise ValueError(
            f"Material '{material_id}' has an empty space_group."
        )

    lattice = record.get("lattice")

    if not isinstance(lattice, dict):
        raise ValueError(
            f"Material '{material_id}' is missing its lattice definition."
        )

    a = _finite_float(
        lattice.get("a"),
        f"{material_id}.lattice.a",
    )

    b = _finite_float(
        lattice.get("b", 0.0),
        f"{material_id}.lattice.b",
    )

    c = _finite_float(
        lattice.get("c", 0.0),
        f"{material_id}.lattice.c",
    )

    alpha = _finite_float(
        lattice.get("alpha", 0.0),
        f"{material_id}.lattice.alpha",
    )

    beta = _finite_float(
        lattice.get("beta", 0.0),
        f"{material_id}.lattice.beta",
    )

    gamma = _finite_float(
        lattice.get("gamma", 0.0),
        f"{material_id}.lattice.gamma",
    )

    if a <= 0.0:
        raise ValueError(
            f"{material_id}.lattice.a must be greater than zero."
        )

    atom_records = record.get("atoms")

    if not isinstance(atom_records, list) or not atom_records:
        raise ValueError(
            f"Material '{material_id}' must contain at least one atom."
        )

    atoms = tuple(
        _parse_atom_record(
            material_id,
            atom_index,
            atom_record,
        )
        for atom_index, atom_record in enumerate(atom_records)
    )

    if density_override in (None, ""):
        density_override = None
    else:
        density_override = _finite_float(
            density_override,
            "density_override",
        )

        if density_override <= 0.0:
            raise ValueError(
                "density_override must be greater than zero."
            )

    return MaterialDefinition(
        id=material_id,
        name=name,
        formula=formula,
        space_group=space_group,
        a=a,
        b=b,
        c=c,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        atoms=atoms,
        density_override_g_cm3=density_override,
    )


def _parse_atom_record(
    material_id: str,
    atom_index: int,
    record: Any,
) -> AtomDefinition:

    prefix = f"{material_id}.atoms[{atom_index}]"

    if not isinstance(record, dict):
        raise ValueError(
            f"{prefix} must be an object."
        )

    label = record.get("label")

    if not isinstance(label, str) or not label.strip():
        raise ValueError(
            f"{prefix}.label must be a non-empty string."
        )

    label = label.strip()

    b_coherent = _finite_float(
        record.get("b_coherent_fm"),
        f"{prefix}.b_coherent_fm",
    )

    sigma_incoherent = _finite_float(
        record.get("sigma_incoherent_barn"),
        f"{prefix}.sigma_incoherent_barn",
    )

    sigma_absorption = _finite_float(
        record.get("sigma_absorption_2200_barn"),
        f"{prefix}.sigma_absorption_2200_barn",
    )

    molar_mass = _finite_float(
        record.get("molar_mass_g_mol"),
        f"{prefix}.molar_mass_g_mol",
    )

    debye_temperature = _finite_float(
        record.get("debye_temperature_k"),
        f"{prefix}.debye_temperature_k",
    )

    x = _finite_float(
        record.get("x"),
        f"{prefix}.x",
    )

    y = _finite_float(
        record.get("y"),
        f"{prefix}.y",
    )

    z = _finite_float(
        record.get("z"),
        f"{prefix}.z",
    )

    raw_positions = record.get("positions")

    if not isinstance(raw_positions, list) or not raw_positions:
        raise ValueError(
            f"{prefix}.positions must contain at least one "
            f"expanded atomic position."
        )

    positions: list[
        tuple[float, float, float]
    ] = []

    for position_index, position in enumerate(raw_positions):
        if (
            not isinstance(position, (list, tuple))
            or len(position) != 3
        ):
            raise ValueError(
                f"{prefix}.positions[{position_index}] "
                f"must contain exactly x, y and z."
            )

        px = _finite_float(
            position[0],
            f"{prefix}.positions[{position_index}][0]",
        )

        py = _finite_float(
            position[1],
            f"{prefix}.positions[{position_index}][1]",
        )

        pz = _finite_float(
            position[2],
            f"{prefix}.positions[{position_index}][2]",
        )

        positions.append(
            (px, py, pz)
        )

    if sigma_incoherent < 0.0:
        raise ValueError(
            f"{prefix}.sigma_incoherent_barn cannot be negative."
        )

    if sigma_absorption < 0.0:
        raise ValueError(
            f"{prefix}.sigma_absorption_2200_barn cannot be negative."
        )

    if molar_mass <= 0.0:
        raise ValueError(
            f"{prefix}.molar_mass_g_mol must be greater than zero."
        )

    if debye_temperature < 0.0:
        raise ValueError(
            f"{prefix}.debye_temperature_k cannot be negative."
        )

    return AtomDefinition(
        label=label,
        b_coherent_fm=b_coherent,
        sigma_incoherent_barn=sigma_incoherent,
        sigma_absorption_2200_barn=sigma_absorption,
        molar_mass_g_mol=molar_mass,
        debye_temperature_k=debye_temperature,
        x=x,
        y=y,
        z=z,
        positions=tuple(positions),
    )


def _finite_float(
    value: Any,
    field_name: str,
) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{field_name} must be numeric."
        ) from exc

    if not math.isfinite(result):
        raise ValueError(
            f"{field_name} must be finite."
        )

    return result
