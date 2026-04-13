# Using DiagnoseHarmonisation in the Terminal

While the ideal way we would recommend using this package is within a python script, we do offer some usage of accessing the cross-sectional reporting tool through the terminal.

After downloading the package and ensuring it is on your python path (check using which python), users are able to run the command 'harmdiag run' in the terminal in order to generate a report.

## Options

The options for running this are seen below:

    "harmdiag",  description="Harmonisation Diagnostics CLI — run harmonisation/reporting from the terminal."
  

    "run", help="Run the diagnostics pipeline from data and covariates CSVs"
    "--data", "-d", required=True, help="Path to data CSV (subjects x IDPs). First row must be feature names."
    "--covariates", "-c", required=True, help="Path to covariates CSV (first column subject ID)."
    "--batch-col", type=int, default=None, help="1-based column number in covariates CSV where batch is located. If omitted, tries to auto-detect by header."
    "--data-id-col", default=None, help="Data subject ID column name (defaults to first column)."
    "--cov-id-col", default=None, help="Covariates subject ID column name (defaults to first column)."
    "--outdir", default=None, help="Directory to write summary / report files."
    "-v", "--verbose", action="store_true", help="Verbose output."
    "--report-name", default=None, help="Optional name for the report (used in filenames)."
    "--save-data", default = True, help="Whether to save the aligned data and covariates used for the report (for debugging)."
    "--save-data-name", default=None, help="Optional name for the saved data files (used in filenames)."

## Notes

We offer some support for different spreedsheet types (e.g xlsx) as well as some support for missing values. However, it is worth noting that if this missingness is relatively high, the pipeline will fail to run (specifically when trying to fit linear mixed effect mdoels). This is true for both data and covariates.

As such we recommend that users use their own imputation approaches or ommit features with large portions of missingnes (>10%). The imputation we do is batch specific, so if batches are small it becomes more unreliable.