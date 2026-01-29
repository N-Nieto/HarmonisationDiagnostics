from __future__ import annotations
import os, re, json, argparse
from typing import List, Dict, Tuple, Any
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager

def test_longitudinal_diagnostic_report():
    import numpy as np
    import pandas as pd
    from scipy.stats import rankdata
    from typing import Sequence, Optional

    from DiagnoseHarmonization import DiagnosticReport
    from DiagnoseHarmonization import DiagnosticFunctions
    from DiagnoseHarmonization import PlotDiagnosticResults

    # fake data
    subjects = ["S1","S2","S3","S1","S2","S3","S1","S2","S3"]
    timepoints = ["T1","T1","T1","T2","T2","T2","T3","T3","T3"]
    batch = [1,1,1,1,1,1,1,1,1]
    age = [50,60,45,23,56,66,33,43,66]

    idp_matrix = np.array([
        [1.2, 0.5],
        [0.9, 0.2],
        [1.5, 0.7],
        [1.3, 0.6],
        [1.0, 0.1],
        [1.6, 0.8],
        [1.5, 0.7],
        [1.3, 0.6],
        [1.0, 0.1],
    ])

    idp_names = ["IDP_A", "IDP_B"]

    results = DiagnosticFunctions.evaluate_pairwise_spearman(
        idp_matrix=idp_matrix,
        subjects=subjects,
        timepoints=timepoints,
        idp_names=idp_names,
        nPerm=1000,
        seed=0,
    )

    print(results)

    ## PLOT
    all_results = [("subjectconsistency", {"pairwise_spearman": results})]   
    outdir      = os.getcwd() 
    figs = PlotDiagnosticResults.plot_pairwise_spearman_combined(all_results, outdir)
    # Save + close manually:

    for label, fig in figs:
        safe_name = "".join(c if (c.isalnum() or c in (' ', '-', '_')) else '_' for c in label).strip().replace(" ", "_")
        fig.savefig(os.path.join(outdir, f"{safe_name}.png"), dpi=200, bbox_inches="tight")
        plt.close(fig)
    # Failed run

    DiagnosticReport.LongitudinalReport(data=idp_matrix, 
                    batch=batch, 
                    subject_ids=subjects, 
                    timepoints=timepoints, 
                    features=idp_names, 
                    covariates=age,
                    covariate_names=["age"],
                    save_data=None,
                    save_dir=None,
                    report_name=None,
                    SaveArtifacts=False,
                    rep= None,
                    show=False,
                    timestamped_reports=True
                    )   

    print("Success!")
    assert True