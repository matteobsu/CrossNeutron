/* ================================================================
   CrossNeutron / BeamMatter frontend
================================================================ */

const MAX_MATERIALS = 8;

let materialLibrary = [];
let latestResult = null;


/* ================================================================
   Startup
================================================================ */

document.addEventListener("DOMContentLoaded", async () => {

  bindControls();

  initializeEmptyPlot();

  await loadMaterialLibrary();

  addMaterialRow("Fe");

});


/* ================================================================
   Controls
================================================================ */

function bindControls() {

  document
    .getElementById("addMaterialBtn")
    .addEventListener("click", () => {
      addMaterialRow();
    });


  document
    .getElementById("calculateBtn")
    .addEventListener("click", calculateAndPlot);


  document
    .getElementById("exportCsvBtn")
    .addEventListener("click", exportCSV);


  document
    .querySelectorAll('input[name="xAxis"]')
    .forEach(radio => {

      radio.addEventListener("change", () => {

        if (latestResult) {
          renderPlot(latestResult);
        }

      });

    });

  document
  .querySelectorAll('input[name="yAxis"]')
  .forEach(radio => {

    radio.addEventListener("change", () => {

      if (latestResult) {
        renderPlot(latestResult);
      }

    });

  });  

}


/* ================================================================
   Material library
================================================================ */

async function loadMaterialLibrary() {

  const response = await fetch("/api/materials");

  if (!response.ok) {

    const message = await response.text();

    throw new Error(
      `Unable to load material library: ${message}`
    );

  }

  const result = await response.json();

  materialLibrary = result.materials || [];

  if (materialLibrary.length === 0) {
    throw new Error("Material library is empty.");
  }

}


function defaultMaterialId() {

  const iron = materialLibrary.find(
    item => item.id === "Fe"
  );

  if (iron) {
    return iron.id;
  }

  return materialLibrary[0].id;

}


/* ================================================================
   Material rows
================================================================ */

function addMaterialRow(selectedId = null) {

  const tbody =
    document.querySelector("#materialTable tbody");

  const currentRows =
    tbody.querySelectorAll("tr");


  if (currentRows.length >= MAX_MATERIALS) {

    alert(
      `A maximum of ${MAX_MATERIALS} materials can be compared.`
    );

    return;

  }


  if (!selectedId) {

    if (currentRows.length > 0) {

      const previous =
        currentRows[
          currentRows.length - 1
        ].querySelector(".material-select");

      selectedId = previous.value;

    } else {

      selectedId = defaultMaterialId();

    }

  }


  const row =
    document.createElement("tr");


  const options =
    materialLibrary
      .map(material => {

        const selected =
          material.id === selectedId
            ? "selected"
            : "";

        const formula =
          material.formula
            ? ` — ${material.formula}`
            : "";

        return `
          <option
            value="${material.id}"
            ${selected}
          >
            ${material.name}${formula}
          </option>
        `;

      })
      .join("");


  row.innerHTML = `

    <td class="material-number"></td>

    <td>
      <button
        type="button"
        class="remove-material"
      >
        ×
      </button>
    </td>

    <td>
      <select class="material-select">
        ${options}
      </select>
    </td>

    <td>
      <input
        type="number"
        class="density-input"
        min="0"
        step="any"
        placeholder="Calculated"
      >
    </td>

  `;


  row
    .querySelector(".remove-material")
    .addEventListener("click", () => {

      const rows =
        tbody.querySelectorAll("tr");

      if (rows.length <= 1) {

        alert(
          "At least one material is required."
        );

        return;

      }

      row.remove();

      renumberMaterials();

    });


  tbody.appendChild(row);

  renumberMaterials();

}


function renumberMaterials() {

  document
    .querySelectorAll("#materialTable tbody tr")
    .forEach((row, index) => {

      row
        .querySelector(".material-number")
        .textContent =
          index + 1;

    });

}


/* ================================================================
   Request
================================================================ */

function buildRequest() {

  const minWavelength =
    Number(
      document.getElementById("minWavelength").value
    );

  const maxWavelength =
    Number(
      document.getElementById("maxWavelength").value
    );

  const points =
    Number(
      document.getElementById("numPoints").value
    );

  const temperature =
    Number(
      document.getElementById("temperature").value
    );


  if (
    !Number.isFinite(minWavelength)
    || minWavelength <= 0
  ) {
    throw new Error(
      "Minimum wavelength must be greater than zero."
    );
  }


  if (
    !Number.isFinite(maxWavelength)
    || maxWavelength <= minWavelength
  ) {
    throw new Error(
      "Maximum wavelength must be greater than minimum wavelength."
    );
  }


  if (
    !Number.isInteger(points)
    || points < 2
    || points > 20000
  ) {
    throw new Error(
      "Calculation points must be an integer between 2 and 20000."
    );
  }


  if (
    !Number.isFinite(temperature)
    || temperature <= 0
  ) {
    throw new Error(
      "Temperature must be greater than zero."
    );
  }


  const materials =
    Array
      .from(
        document.querySelectorAll(
          "#materialTable tbody tr"
        )
      )
      .map(row => {

        const id =
          row
            .querySelector(".material-select")
            .value;

        const densityText =
          row
            .querySelector(".density-input")
            .value
            .trim();


        const item = {
          id: id
        };


        if (densityText !== "") {

          const density =
            Number(densityText);

          if (
            !Number.isFinite(density)
            || density <= 0
          ) {
            throw new Error(
              `Invalid density for ${id}.`
            );
          }

          item.density_g_cm3 =
            density;

        }


        return item;

      });


  const components =
    Array
      .from(
        document.querySelectorAll(
          ".component-option input[type='checkbox']:checked"
        )
      )
      .map(input => input.value);


  if (components.length === 0) {

    throw new Error(
      "Select at least one cross-section component."
    );

  }


  return {

    wavelength: {

      min:
        minWavelength,

      max:
        maxWavelength,

      points:
        points

    },

    temperature_k:
      temperature,

    materials:
      materials,

    components:
      components,

    options: {}

  };

}


/* ================================================================
   Calculate
================================================================ */

async function calculateAndPlot() {

  const button =
    document.getElementById("calculateBtn");

  const status =
    document.getElementById("statusText");


  button.disabled = true;

  status.textContent =
    "Calculating...";


  try {

    const payload =
      buildRequest();


    const response =
      await fetch(
        "/api/calculate",
        {

          method: "POST",

          headers: {
            "Content-Type": "application/json"
          },

          body:
            JSON.stringify(payload)

        }
      );


    const result =
      await response.json();


    if (!response.ok) {

      throw new Error(
        result.error || "Calculation failed."
      );

    }


    latestResult =
      result;


    renderPlot(
      result
    );


    renderMaterialProperties(
      result.materials || []
    );


    document
      .getElementById("exportCsvBtn")
      .disabled =
        false;


    document
      .getElementById("plotMessage")
      .textContent =
        "";


    status.textContent =
      "Finished";


  } catch (error) {

    console.error(error);

    status.textContent =
      "Error";

    alert(
      error.message
    );

  } finally {

    button.disabled =
      false;

  }

}


/* ================================================================
   Plot
================================================================ */

function getXAxisMode() {

  return (
    document.querySelector(
      'input[name="xAxis"]:checked'
    )?.value
    || "wavelength"
  );

}

function getYAxisMode() {

  return (
    document.querySelector(
      'input[name="yAxis"]:checked'
    )?.value
    || "attenuation"
  );

}

function initializeEmptyPlot() {

  Plotly.newPlot(

    "crossSectionPlot",

    [],

    {

      xaxis: {
        title: "Wavelength λ [Å]"
      },

      yaxis: {
        title: "Linear attenuation coefficient μ [cm⁻¹]"
      },

      annotations: [
        {
          text:
            "Select a material and press Calculate / Plot",

          x: 0.5,
          y: 0.5,

          xref: "paper",
          yref: "paper",

          showarrow: false
        }
      ]

    },

    {
      responsive: true,
      displaylogo: false
    }

  );

}


function renderPlot(result) {

  const mode =
    getXAxisMode();


  const yMode =
    getYAxisMode();


  const energyMode =
    mode === "energy";


  const attenuationMode =
    yMode === "attenuation";


  const energyMode =
    mode === "energy";


  const rawX =
    energyMode
      ? result.axes.energy_mev
      : result.axes.wavelength_a;


  const xTitle =
    energyMode
      ? "Neutron energy E [meV]"
      : "Wavelength λ [Å]";

  const yTitle =
    attenuationMode
      ? "Linear attenuation coefficient μ [cm⁻¹]"
      : "Cross section σ [barn]";


  const yHoverLabel =
    attenuationMode
      ? "μ: %{y:.5g} cm⁻¹"
      : "σ: %{y:.5g} barn";


  const traces =
    result.series.map(series => {

      let x =
        rawX;

      let y =
        attenuationMode
          ? series.values_cm_inv
          : series.values_barn;


      /*
       * Wavelength increases while neutron energy decreases.
       * Reverse both arrays so the energy plot runs naturally
       * from low to high energy.
       */

      if (energyMode) {

        x =
          [...x].reverse();

        y =
          [...y].reverse();

      }


      return {

        x: x,

        y: y,

        type: "scatter",

        mode: "lines",

        name:
          series.name,

        hovertemplate:

          `${xTitle}: %{x:.5g}<br>` +
          yHoverLabel +
          "<extra>%{fullData.name}</extra>"

      };

    });


  Plotly.react(

    "crossSectionPlot",

    traces,

    {

      margin: {
        l: 75,
        r: 25,
        t: 30,
        b: 80
      },

      xaxis: {

        title:
          xTitle,

        automargin:
          true

      },

      yaxis: {

        title:
          yTitle,

        rangemode:
          "tozero",

        automargin:
          true

      },

      hovermode:
        "closest",

      legend: {

        orientation:
          "h",

        x:
          0.5,

        xanchor:
          "center",

        y:
          -0.2,

        yanchor:
          "top"

      }

    },

    {

      responsive:
        true,

      displaylogo:
        false

    }

  );

}


/* ================================================================
   Material properties
================================================================ */

function renderMaterialProperties(materials) {

  const tbody =
    document.querySelector(
      "#propertiesTable tbody"
    );


  tbody.innerHTML =
    "";


  if (materials.length === 0) {

    tbody.innerHTML = `

      <tr class="placeholder-row">

        <td colspan="7">
          No material data available.
        </td>

      </tr>

    `;

    return;

  }


  materials.forEach(material => {

    const row =
      document.createElement("tr");


    row.innerHTML = `

      <td>
        ${escapeHtml(material.name)}
      </td>

      <td>
        ${formatValue(material.n_atoms_unit_cell)}
      </td>

      <td>
        ${formatValue(material.volume_a3)}
      </td>

      <td>
        ${formatValue(material.mass_g_mol_unit_cell)}
      </td>

      <td>
        ${formatValue(material.density_g_cm3)}
      </td>

      <td>
        ${formatValue(material.avg_sigma_coherent_barn)}
      </td>

      <td>
        ${formatValue(material.avg_sigma_incoherent_barn)}
      </td>

    `;


    tbody.appendChild(row);

  });

}


function formatValue(value) {

  if (
    value === null
    || value === undefined
  ) {
    return "–";
  }


  if (
    typeof value === "number"
  ) {

    return Number(
      value.toPrecision(7)
    );

  }


  return value;

}


function escapeHtml(value) {

  const element =
    document.createElement("div");

  element.textContent =
    value ?? "";

  return element.innerHTML;

}


/* ================================================================
   CSV
================================================================ */

function exportCSV() {

  if (!latestResult) {

    alert(
      "Run a calculation first."
    );

    return;

  }

  const attenuationMode =
    getYAxisMode() === "attenuation";


  const unitLabel =
    attenuationMode
      ? "cm^-1"
      : "barn";


  const valueKey =
    attenuationMode
      ? "values_cm_inv"
      : "values_barn";


  const headers = [

    "Wavelength [A]",

    "Energy [meV]",

    ...latestResult.series.map(
      series =>
        `${series.name} [${unitLabel}]`
    )

  ];


  const rows = [
    headers
  ];


  const length =
    latestResult.axes.wavelength_a.length;


  for (
    let index = 0;
    index < length;
    index++
  ) {

    rows.push([

      latestResult
        .axes
        .wavelength_a[index],

      latestResult
        .axes
        .energy_mev[index],

      ...latestResult
        .series
        .map(
          series =>
            series[valueKey][index]
        )

    ]);

  }


  const csv =
    rows
      .map(row =>

        row
          .map(value => {

            const escaped =
              String(value)
                .replace(
                  /"/g,
                  '""'
                );

            return `"${escaped}"`;

          })
          .join(";")

      )
      .join("\n");


  const blob =
    new Blob(

      [
        "\uFEFF" + csv
      ],

      {
        type:
          "text/csv;charset=utf-8;"
      }

    );


  const link =
    document.createElement("a");


  link.href =
    URL.createObjectURL(blob);

  link.download =
    attenuationMode
      ? "neutron_linear_attenuation.csv"
      : "neutron_cross_sections.csv";


  document.body.appendChild(
    link
  );

  link.click();

  link.remove();


  URL.revokeObjectURL(
    link.href
  );

}