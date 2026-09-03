"""
BeamMatter
Neutron cross-section web application.

This file is only the HTTP/API layer.

It does NOT contain:
    - neutron cross-section equations
    - crystallography calculations
    - SGInfo logic
    - reflection generation
    - material constants

Those responsibilities belong to:
    calculate.py
    materials.py
    symmetry/crystallography backend
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request, send_from_directory

from calculate import calculate_cross_sections
from materials import get_material_catalog, resolve_material


# ======================================================================
# Paths
# ======================================================================

BASE_DIR = Path(__file__).resolve().parent


# ======================================================================
# Application
# ======================================================================

app = Flask(
    __name__,
    static_folder=None,
)

app.config["JSON_SORT_KEYS"] = False


# ======================================================================
# Limits
# ======================================================================

MAX_MATERIALS = 8
MAX_CALCULATION_POINTS = 20000

DEFAULT_TEMPERATURE_K = 293.0

DEFAULT_WAVELENGTH_MIN_A = 0.1
DEFAULT_WAVELENGTH_MAX_A = 10.0
DEFAULT_POINTS = 1000


# ======================================================================
# Frontend
# ======================================================================

@app.get("/")
def index():
    """
    Serve the main web interface.
    """

    return send_from_directory(
        BASE_DIR,
        "index.html",
    )


@app.get("/script.js")
def script():
    return send_from_directory(
        BASE_DIR,
        "script.js",
    )


@app.get("/style.css")
def style():
    return send_from_directory(
        BASE_DIR,
        "style.css",
    )


@app.get("/assets/<path:filename>")
def assets(filename: str):
    """
    Optional location for images, logos, icons, etc.

    Project structure:

        assets/
            logo.png
            ...
    """

    return send_from_directory(
        BASE_DIR / "assets",
        filename,
    )


# ======================================================================
# Health
# ======================================================================

@app.get("/api/health")
def health():
    """
    Lightweight health check for local use and Render.
    """

    return jsonify({
        "status": "ok",
        "application": "BeamMatter",
    })


# ======================================================================
# Material library
# ======================================================================

@app.get("/api/materials")
def materials():
    """
    Return the lightweight material list used by the frontend.

    Example response:

    {
        "materials": [
            {
                "id": "Fe",
                "name": "Iron",
                "formula": "Fe"
            },
            ...
        ]
    }
    """

    try:

        catalog = get_material_catalog()

        return jsonify({
            "materials": catalog,
        })

    except Exception:

        app.logger.exception(
            "Failed to load material catalog."
        )

        return jsonify({
            "error": "Unable to load material library.",
        }), 500


# ======================================================================
# Main calculation
# ======================================================================

@app.post("/api/calculate")
def calculate():
    """
    Calculate neutron cross sections.

    Expected request:

    {
        "wavelength": {
            "min": 0.1,
            "max": 10.0,
            "points": 1000
        },

        "temperature_k": 293.0,

        "materials": [
            {
                "id": "Fe"
            },
            {
                "id": "Al"
            }
        ],

        "components": [
            "total",
            "coherent_elastic",
            "incoherent_elastic",
            "coherent_inelastic",
            "incoherent_inelastic",
            "absorption"
        ],

        "options": {
            "max_hkl": null
        }
    }

    Density may optionally be overridden:

        {
            "id": "Fe",
            "density_g_cm3": 7.87
        }

    Normally density will come from the material/crystal calculation.
    """

    # ------------------------------------------------------------------
    # JSON
    # ------------------------------------------------------------------

    data = request.get_json(
        silent=True,
    )

    if not isinstance(data, dict):

        return api_error(
            "Request body must be a JSON object."
        )


    # ------------------------------------------------------------------
    # Wavelength
    # ------------------------------------------------------------------

    try:

        wavelength = parse_wavelength_settings(
            data.get("wavelength")
        )

    except ValueError as exc:

        return api_error(
            str(exc)
        )


    # ------------------------------------------------------------------
    # Temperature
    # ------------------------------------------------------------------

    try:

        temperature_k = parse_positive_float(
            data.get(
                "temperature_k",
                DEFAULT_TEMPERATURE_K,
            ),
            "temperature_k",
        )

    except ValueError as exc:

        return api_error(
            str(exc)
        )


    # ------------------------------------------------------------------
    # Components
    # ------------------------------------------------------------------

    try:

        components = parse_components(
            data.get("components")
        )

    except ValueError as exc:

        return api_error(
            str(exc)
        )


    # ------------------------------------------------------------------
    # Materials
    # ------------------------------------------------------------------

    try:

        materials = parse_materials(
            data.get("materials")
        )

    except ValueError as exc:

        return api_error(
            str(exc)
        )


    # ------------------------------------------------------------------
    # Advanced options
    # ------------------------------------------------------------------

    try:

        options = parse_options(
            data.get("options")
        )

    except ValueError as exc:

        return api_error(
            str(exc)
        )


    # ------------------------------------------------------------------
    # Resolve materials
    # ------------------------------------------------------------------

    resolved_materials = []


    try:

        for item in materials:

            material = resolve_material(

                material_id=
                    item["id"],

                density_override=
                    item.get(
                        "density_g_cm3"
                    ),

            )

            resolved_materials.append(
                material
            )


    except KeyError as exc:

        return api_error(
            str(exc),
            status=404,
        )


    except ValueError as exc:

        return api_error(
            str(exc)
        )


    # ------------------------------------------------------------------
    # Physics
    # ------------------------------------------------------------------

    try:

        result = calculate_cross_sections(

            materials=
                resolved_materials,

            wavelength_min_a=
                wavelength["min"],

            wavelength_max_a=
                wavelength["max"],

            points=
                wavelength["points"],

            temperature_k=
                temperature_k,

            components=
                components,

            max_hkl=
                options["max_hkl"],

        )


    except ValueError as exc:

        return api_error(
            str(exc)
        )


    except Exception:

        app.logger.exception(
            "Neutron cross-section calculation failed."
        )

        return jsonify({
            "error": "Calculation failed.",
        }), 500


    # ------------------------------------------------------------------
    # Result
    # ------------------------------------------------------------------

    return jsonify({
        "status": "ok",
        **result,
    })


# ======================================================================
# Request parsing
# ======================================================================

def parse_wavelength_settings(
    value: Any,
) -> dict[str, Any]:
    """
    Parse wavelength settings.

    Returned wavelengths are always in Angstrom.
    """

    if value is None:

        value = {}


    if not isinstance(value, dict):

        raise ValueError(
            "'wavelength' must be an object."
        )


    minimum = parse_positive_float(
        value.get(
            "min",
            DEFAULT_WAVELENGTH_MIN_A,
        ),
        "wavelength.min",
    )


    maximum = parse_positive_float(
        value.get(
            "max",
            DEFAULT_WAVELENGTH_MAX_A,
        ),
        "wavelength.max",
    )


    if maximum <= minimum:

        raise ValueError(
            "wavelength.max must be greater "
            "than wavelength.min."
        )


    raw_points = value.get(
        "points",
        DEFAULT_POINTS,
    )


    try:

        points = int(
            raw_points
        )

    except (
        TypeError,
        ValueError,
    ) as exc:

        raise ValueError(
            "wavelength.points must be an integer."
        ) from exc


    if points < 2:

        raise ValueError(
            "wavelength.points must be at least 2."
        )


    if points > MAX_CALCULATION_POINTS:

        raise ValueError(
            f"wavelength.points cannot exceed "
            f"{MAX_CALCULATION_POINTS}."
        )


    return {
        "min": minimum,
        "max": maximum,
        "points": points,
    }


# ======================================================================
# Components
# ======================================================================

ALLOWED_COMPONENTS = {

    "total",

    "coherent_elastic",

    "incoherent_elastic",

    "coherent_inelastic",

    "incoherent_inelastic",

    "total_inelastic",

    "absorption",

}


DEFAULT_COMPONENTS = [

    "total",

    "coherent_elastic",

    "incoherent_elastic",

    "coherent_inelastic",

    "incoherent_inelastic",

    "absorption",

]


def parse_components(
    value: Any,
) -> list[str]:
    """
    Validate requested NXS calculation components.

    Names correspond directly to quantities available from libnxs:

        nxs_CoherentElastic
        nxs_IncoherentElastic
        nxs_CoherentInelastic
        nxs_IncoherentInelastic
        nxs_TotalInelastic
        nxs_Absorption

    'total' is the final combined cross section calculated by
    calculate.py.
    """

    if value is None:

        return list(
            DEFAULT_COMPONENTS
        )


    if not isinstance(value, list):

        raise ValueError(
            "'components' must be an array."
        )


    if not value:

        raise ValueError(
            "At least one cross-section component "
            "must be selected."
        )


    components = []


    for component in value:

        if not isinstance(
            component,
            str,
        ):

            raise ValueError(
                "Cross-section component names "
                "must be strings."
            )


        component = (
            component
            .strip()
            .lower()
        )


        if component not in ALLOWED_COMPONENTS:

            raise ValueError(
                f"Unknown cross-section component: "
                f"'{component}'."
            )


        if component not in components:

            components.append(
                component
            )


    return components


# ======================================================================
# Materials
# ======================================================================

def parse_materials(
    value: Any,
) -> list[dict[str, Any]]:
    """
    Validate selected materials.

    Input:

        [
            {
                "id": "Fe"
            },

            {
                "id": "Al",
                "density_g_cm3": 2.70
            }
        ]
    """

    if not isinstance(
        value,
        list,
    ):

        raise ValueError(
            "'materials' must be an array."
        )


    if not value:

        raise ValueError(
            "At least one material must be selected."
        )


    if len(value) > MAX_MATERIALS:

        raise ValueError(
            f"A maximum of {MAX_MATERIALS} materials "
            f"can be calculated at once."
        )


    materials = []


    for index, item in enumerate(
        value,
        start=1,
    ):

        if not isinstance(
            item,
            dict,
        ):

            raise ValueError(
                f"materials[{index - 1}] "
                f"must be an object."
            )


        material_id = item.get(
            "id"
        )


        if not isinstance(
            material_id,
            str,
        ):

            raise ValueError(
                f"materials[{index - 1}].id "
                f"must be a string."
            )


        material_id = (
            material_id
            .strip()
        )


        if not material_id:

            raise ValueError(
                f"materials[{index - 1}].id "
                f"cannot be empty."
            )


        density = item.get(
            "density_g_cm3"
        )


        if density in (
            None,
            "",
        ):

            density = None

        else:

            density = parse_positive_float(
                density,
                (
                    f"materials[{index - 1}]"
                    ".density_g_cm3"
                ),
            )


        materials.append({

            "id":
                material_id,

            "density_g_cm3":
                density,

        })


    return materials


# ======================================================================
# Advanced calculation options
# ======================================================================

def parse_options(
    value: Any,
) -> dict[str, Any]:
    """
    Parse calculation-engine options.

    max_hkl is retained as an optional advanced setting because the
    original NXS implementation stores maxHKL_index in NXS_UnitCell
    and uses it during nxs_initHKL().

    The final calculate.py can choose its own automatic/default value
    whenever this is None.
    """

    if value is None:

        value = {}


    if not isinstance(
        value,
        dict,
    ):

        raise ValueError(
            "'options' must be an object."
        )


    max_hkl = value.get(
        "max_hkl"
    )


    if max_hkl in (
        None,
        "",
    ):

        max_hkl = None

    else:

        try:

            max_hkl = int(
                max_hkl
            )

        except (
            TypeError,
            ValueError,
        ) as exc:

            raise ValueError(
                "options.max_hkl must be an integer."
            ) from exc


        if max_hkl < 1:

            raise ValueError(
                "options.max_hkl must be greater than zero."
            )


        if max_hkl > 200:

            raise ValueError(
                "options.max_hkl is unreasonably large."
            )


    return {
        "max_hkl": max_hkl,
    }


# ======================================================================
# Generic value parsing
# ======================================================================

def parse_positive_float(
    value: Any,
    field_name: str,
) -> float:

    try:

        result = float(
            value
        )

    except (
        TypeError,
        ValueError,
    ) as exc:

        raise ValueError(
            f"{field_name} must be numeric."
        ) from exc


    if result <= 0:

        raise ValueError(
            f"{field_name} must be greater than zero."
        )


    return result


# ======================================================================
# API errors
# ======================================================================

def api_error(
    message: str,
    status: int = 400,
):

    return jsonify({
        "error": message,
    }), status


# ======================================================================
# Flask errors
# ======================================================================

@app.errorhandler(404)
def not_found(_error):

    return jsonify({
        "error": "Not found.",
    }), 404


@app.errorhandler(405)
def method_not_allowed(_error):

    return jsonify({
        "error": "Method not allowed.",
    }), 405


# ======================================================================
# Development server
# ======================================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            "5000",
        )
    )


    debug = (
        os.environ.get(
            "FLASK_DEBUG",
            ""
        )
        .strip()
        .lower()
        in {
            "1",
            "true",
            "yes",
            "on",
        }
    )


    app.run(
        host="0.0.0.0",
        port=port,
        debug=debug,
    )