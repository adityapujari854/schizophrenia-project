# Schizophrenia Research Project

An exploratory research prototype for schizophrenia-related neuroimaging and gene-expression analysis. It combines Python data-science workflows with a TypeScript/React web interface for submitting data and viewing analysis results.

## Project structure

- `components/` and `pages/` contain the web interface.
- `schizo_project/scripts/` contains Python workflows for preprocessing, feature engineering, dimensionality reduction, model experiments, reporting, and visualization.
- `schizo_project/configs/` contains experiment-grid configuration files.
- `schizo_project/assets/` contains image assets used by the research app.

## Data and generated files

Datasets, uploaded scans, trained models, caches, plots, generated results, and backups are intentionally excluded. Obtain data from its original source, follow applicable licenses and research-ethics requirements, and place it in the local paths expected by the scripts. The source snapshot does not include dependency manifests, so runtime setup may require additional project-specific configuration.

## Disclaimer

This project is for research and educational use only. It is not a medical device and must not be used for clinical diagnosis or treatment.
