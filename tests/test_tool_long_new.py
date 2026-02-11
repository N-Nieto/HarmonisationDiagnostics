"""Simple test script to check new structure of longitudinal_report_experimental.py"""


import os
import numpy as np
from pathlib import Path
import pytest
import time
import webbrowser

# Adjust import to match your package layout
# from DiagnoseHarmonization.DiagnosticReport import DiagnosticReport
# If DiagnosticReport is defined in a module file DiagnosticReport.py inside DiagnoseHarmonization package:

save_dir = "/Users/jacob.turnbull/VS_code_projects/diagnostic_full_run/"

def test_full_pipeline_generates_report(tmp_path = save_dir):
    """
    Run the full DiagnosticReport pipeline once and produce a single HTML report.
    - Writes report to a temporary directory (tmp_path / "diagnostic_full_run")
    - Asserts that a timestamped DiagnosticReport_*.html file was created
    - Prints the path to the report for manual viewing
    - Optionally opens the report automatically when OPEN_REPORT=1
    """
    import numpy as np
    import pandas as pd

    from DiagnoseHarmonization import DiagnosticFunctions
    from DiagnoseHarmonization import PlotDiagnosticResults
    from DiagnoseHarmonization import DiagnosticReport

    # Load CSV
    df = pd.read_csv("tests/onharmony.csv")

    # ---- REQUIRED STRUCTURAL VARIABLES ----
    subjects   = df["subject"].astype(str).tolist()
    timepoints = df["timepoint"].astype(str).tolist()
    batches    = df["scan_session"].astype(str).tolist()
    age = df["age"].astype(int).tolist()
    sex = df["sex"].astype(str).tolist()
    print(age)
    # ---- COVARIATES (dict) ----
    covariates = {
        "age": age,
        "sex": sex
    }
    # Check max age large than 50 to ensure it's not being read as a string
    assert max(age) > 45, "Max age should be greater than 50, check if age is being read as string instead of numeric."
    covariates_array = np.column_stack([age, sex])
    #print(covariates_array)

    # ---- IDPs ----
    idp_names = ["T1_SIENAX_peripheral_GM_norm_vol", "T1_SIENAX_WM_norm_vol"]   # or infer automatically (see below)
    idp_matrix = df[idp_names].to_numpy(dtype=float)

    # ---- SANITY CHECKS ----
    n_samples = len(df)
    print("n_samples:", n_samples)
    print("shape idp_matrix:", idp_matrix.shape)
    print("subjects (unique):", sorted(set(subjects)))
    print("batches (unique):", sorted(set(batches)))

    DiagnosticReport.longitudinal_report_experimental(data=idp_matrix, 
                                                      subject_ids=subjects, 
                                                      batch=batches, 
                                                      timepoints=timepoints, 
                                                      features=idp_names, 
                                                      covariates=covariates,
                                                      covariate_names=list(covariates.keys()),
                                                      save_dir=tmp_path,
                                                      save_data=True,
                                                      report_name="longitudinal_report_test.html",
                                                      timestamped_reports=False)  # Set to True to automatically open the report
    

