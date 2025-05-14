# LBNL\_PPC01 Repository Overview

## Description

The `LBNL_PPC01` repository is a collection of scripts designed to process and analyze data from the LBNL PPC01 test stand. All processing workflows rely on the [Juleanita](https://github.com/LisaSchlueter/Juleanita.jl) package, with supporting data and metadata management handled via [LegendDataManagement](https://github.com/legend-exp/LegendDataManagement) and follows the standard LEGEND processing methods.

## Folder Structure

The repository is organized into two primary data categories, each with a consistent internal structure:

```
LBNL_PPC01/
├── cal/
│   ├── standard_dataflow/     # Full data processing using Juleanita processors
│   ├── standard_plots/        # Post-processing plots
│   └── scripts/               # Custom scripts for specific calibration studies
├── bch/
│   ├── standard_dataflow/     # Full data processing using Juleanita processors
│   ├── standard_plots/        # Post-processing plots
│   └── scripts/               # Custom scripts for specific bechtest runs
```

* **`cal/`**: Contains scripts and workflows for calibration data.
* **`bch/`**: Contains scripts and workflows for bechtest data.

Each of these main folders (`cal`, `bch`) includes:

* `standard_dataflow`: Core processing pipelines using Juleanita.
* `standard_plots`: Scripts to generate visualizations from processed data.
* `scripts`: Additional tools and analyses tailored for specific periods or runs.

## Metadata and configs

All configs and metadata required for data processing are maintained in a separate repository:
➡️ [teststand-metadata](https://github.com/LisaSchlueter/teststand-metadata). 

