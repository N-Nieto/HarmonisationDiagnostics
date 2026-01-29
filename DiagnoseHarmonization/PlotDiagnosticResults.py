#%%


"""----------------------------------------------------------------------------------------------------------------------------"""
"""---------------------------------------- TEST WRAPPER FUNCTION ----------------------------------"""
"""----------------------------------------------------------------------------------------------------------------------------"""
import inspect
from functools import wraps
from typing import Any, Callable, List, Tuple, Optional
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.figure as mfig
from scipy import stats

def _is_figure(obj) -> bool:
    return isinstance(obj, mfig.Figure)

def _normalize_figs_from_result(result: Any) -> List[Tuple[Optional[str], mfig.Figure]]:
    """Normalize many possible return shapes into a list of (caption, Figure)."""
    if result is None:
        return []
    if _is_figure(result):
        return [(None, result)]
    if isinstance(result, tuple) and len(result) >= 1 and _is_figure(result[0]):
        return [(None, result[0])]
    if isinstance(result, (list, tuple)):
        out = []
        for item in result:
            if _is_figure(item):
                out.append((None, item))
            elif isinstance(item, (list, tuple)) and len(item) >= 2 and _is_figure(item[1]):
                out.append((str(item[0]) if item[0] is not None else None, item[1]))
        return out
    if isinstance(result, dict):
        for k in ("fig", "figure", "figures"):
            if k in result:
                return _normalize_figs_from_result(result[k])
    return []

def rep_plot_wrapper(func: Callable) -> Callable:
    """
    Decorator that:
      - optionally forces show=False (if the wrapped function supports it),
      - intercepts and removes wrapper-only kwargs (rep, log_func, caption),
      - logs returned figure(s) into rep via rep.log_plot(fig, caption) if rep provided,
      - closes figures after logging to free memory.
    """
    @wraps(func)
    def _wrapper(*args, **kwargs):
        # Extract wrapper-only args and remove them from kwargs BEFORE calling func
        rep = kwargs.pop("rep", None)
        log_func = kwargs.pop("log_func", None)
        caption_kw = kwargs.pop("caption", None)

        # If function supports 'show', force show=False unless caller explicitly set it
        try:
            sig = inspect.signature(func)
            if "show" in sig.parameters and "show" not in kwargs:
                kwargs["show"] = False
        except Exception:
            pass

        # Call original function without rep/log_func/caption in kwargs
        result = func(*args, **kwargs)

        # If neither rep nor log_func provided, return the original result unchanged
        if rep is None and log_func is None:
            return result

        # Normalize any returned figures
        figs = _normalize_figs_from_result(result)
        if not figs:
            # nothing to log; return original result for backward compatibility
            return result

        # Log each figure (use caption from return value or fallback)
        for idx, (cap, fig) in enumerate(figs):
            used_caption = cap or caption_kw or f"{func.__name__} — plot {idx+1}"
            try:
                if rep is not None:
                    rep.log_plot(fig, used_caption)
                elif callable(log_func):
                    log_func(fig, used_caption)
            except Exception as e:
                # best-effort: if rep has log_text, write the error there
                try:
                    if rep is not None and hasattr(rep, "log_text"):
                        rep.log_text(f"Failed to log figure from {func.__name__}: {e}")
                except Exception:
                    pass
            finally:
                try:
                    plt.close(fig)
                except Exception:
                    pass

        # Return original result (keeps backward compatibility)
        return result

    return _wrapper

#%%

import matplotlib.pyplot as plt
from collections.abc import Sequence
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections.abc import Sequence

"""
    Complementary plotting functions for the functions in DiagnosticFunctions.py

    Functions:
    - Z_Score_Plot: Plot histogram and heatmap of Z-scored data by batch.
    - Cohens_D_plot: Plot Cohen's d effect sizes with histograms.
    - variance_ratio_plot: Plot variance ratios between batches.
    - PC_corr_plot: Generate PCA diagnostic plots including scatter plots and correlation heatmaps.
    - PC_clustering_plot: K-means clustering and silhouette analysis of PCA results by batch.
    - Ks_Plot: Plot KS statistic between batches.
    - 


"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from typing import Optional
import pandas as pd
"""----------------------------------------------------------------------------------------------------------------------------"""
"""---------------------------------------- Plotting functions for Z-score results ----------------------------------"""
"""----------------------------------------------------------------------------------------------------------------------------"""
@rep_plot_wrapper
def Z_Score_Plot(data, batch, probablity_distribution=False,draw_PDF=True):
    """
    Plots the median centered Z-score data as a heatmap and as a histogram of all scores.
    Re-order by batch for better visualisaion in the heatmap, also plot batch seperators on heatmap.
    Args:
        data (np.ndarray): 2D array of Z-scored data (samples x features).
    Returns:
        None: Displays plot of Z-scored data and a histogram of the values on different axes.
    """
    # Histogram of all Z-scores plotted by batch variable
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import numpy as np
    from matplotlib.figure import Figure
    from scipy import stats

    # ---- Validation ----
    if not isinstance(data, np.ndarray):
        raise ValueError("data must be a NumPy array.")
    if data.ndim != 2:
        raise ValueError("data must be a 2D array (samples x features).")
    if not isinstance(batch, np.ndarray):
        if isinstance(batch, list):
            batch = np.array(batch)
        else:
            raise ValueError("batch must be a NumPy array or a list.")
    # Sort data by batch, batch can either be numeric or string labels here:
    sorted_indices = np.argsort(batch)
    sorted_data = data[sorted_indices, :]
    sorted_batch = batch[sorted_indices]
    unique_batches, batch_counts = np.unique(sorted_batch, return_counts=True)  
    # Create figure with gridspec
    fig = plt.figure(figsize=(14, 8))
    # Loop over unique batches and plot as histogram on same axis:
    ax1 = fig.add_subplot()
    import matplotlib
    # Define colours for each histogram based on number of unique batches:
    colors = matplotlib.pyplot.get_cmap('tab10', len(unique_batches))

    if probablity_distribution==True:
        plot_type = 'density'
    else:
        plot_type = 'frequency'

    for i in np.unique(batch):
        batch_data = data[batch == i, :].flatten()
        # Match colours of the histogram for each batch:
        color = colors(np.where(unique_batches == i)[0][0])
        ax1.hist(batch_data, bins=80, density=plot_type, alpha=0.5, label=str(i), color=color)
        # Draw an estimated normal distribution curve over histogram:
        if draw_PDF==True:
            mu, std = np.mean(batch_data), np.std(batch_data)
            xmin, xmax = ax1.get_xlim()
            x = np.linspace(xmin, xmax, 100)
            p = stats.norm.pdf(x, mu, std)            
            ax1.plot(x, p, color=color, linewidth=2)

    ax1.set_xlabel("Z-scores of all unique measures")
    # Set axis limits to -8 to 8 for better visualisation:
    ax1.set_xlim([-8, 8])
    ax1.invert_xaxis()
    ax1.yaxis.tick_right()
    ax1.yaxis.set_label_position("right")
    ax1.legend(title="Batch")

    figs = []
    figs.append(("Z-score histogram", fig))
    #figs.append(("Z-score heatmap", fig2))
    return figs
"""----------------------------------------------------------------------------------------------------------------------------"""
"""---------------------------------------- Plotting functions for Cohens D results ----------------------------------"""
"""----------------------------------------------------------------------------------------------------------------------------"""
@rep_plot_wrapper
def Cohens_D_plot(
    cohens_d: np.ndarray,
    pair_labels: list,
    df: Optional[pd.DataFrame] = None,
    *,
    rep = None,            # optional StatsReporter
    caption: Optional[str] = None,
    show: bool = False
) -> plt.Figure:
    # (validation code unchanged)...
    if not isinstance(cohens_d, np.ndarray):
        raise ValueError("cohens_d must be a NumPy array.")
    if cohens_d.ndim != 2:
        raise ValueError("cohens_d must be a 2D array (num_pairs x num_features).")
    if not isinstance(pair_labels, list) or len(pair_labels) != cohens_d.shape[0]:
        raise ValueError("pair_labels must be a list with the same length as cohens_d rows.")
    
    # Create one figure per pair and return a list or just create+log each inside loop:
    figs = []
    for i in range(cohens_d.shape[0]):
        fig = plt.figure(figsize=(14, 8))
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 8], wspace=0.3)
        ax1 = fig.add_subplot(gs[0])
        ax1.hist(cohens_d[i], bins=20, orientation='horizontal', color=[0.8, 0.2, 0.2])
        ax1.set_xlabel("Frequency")
        ax1.invert_xaxis()
        ax1.yaxis.tick_right()
        ax1.yaxis.set_label_position("right")
        ax2 = fig.add_subplot(gs[1], sharey=ax1)
        indices = np.arange(cohens_d.shape[1])
        ax2.bar(indices, cohens_d[i], color=[0.2, 0.4, 0.6])
        ax2.plot(indices, cohens_d[i], 'r.')
        # add effect size lines...
        ax2.set_xlabel("Feature Index")
        ax2.set_ylabel("Cohen's d")
        ax2.set_title(f"Effect Size (Cohen's d) for {pair_labels[i]}")
        #fig.tight_layout()
        ax2.grid(True)
        # Ensure equal y-limits for fair comparison
        # Draw horizontal lines for small/medium/large effect sizes, green small, orange medium, red large
        for thresh, color, label in [ (0.2, 'green', 'Small'), (0.5, 'orange', 'Medium'), (0.8, 'red', 'Large') ]:
            ax2.axhline(y=thresh, color=color, linestyle='--', linewidth=1)
            ax2.axhline(y=-thresh, color=color, linestyle='--', linewidth=1)
            ax2.text(cohens_d.shape[1]-1, thresh, f' {label}', color=color, va='bottom', ha='right', fontsize=8)
            ax2.text(cohens_d.shape[1]-1, -thresh, f' {label}', color=color, va='top', ha='right', fontsize=8)
        # Set limits to have equal negatice/positive range around zero
        ylims = ax2.get_ylim()
        max_abs = max(abs(ylims[0]), abs(ylims[1]))
        ax2.set_ylim(-max_abs, max_abs)
        ax1.set_ylim(-max_abs, max_abs)


        caption_i = caption or f"Cohen's d — {pair_labels[i]}"
        if rep is not None:
            rep.log_plot(fig, caption_i)
            plt.close(fig)
        else:
            figs.append((caption_i, fig))
            if show:
                plt.show()
    # If rep used, figs list is empty; otherwise return list for caller
    return None if rep is not None else figs
"""----------------------------------------------------------------------------------------------------------------------------"""
"""---------------------------------------- Plotting functions ratio of variance ----------------------------------"""
"""----------------------------------------------------------------------------------------------------------------------------"""
def variance_ratio_plot(variance_ratios:  np.ndarray, pair_labels: list,
                         df: None = None,rep = None,show: bool = False,caption: Optional[str] = None,) -> None:
    """
    Plots the explained variance ratio for each principal component as a bar plot.

    Args:
        variance_ratios (Sequence[float]): A sequence of explained variance ratios for each principal component.
    Returns:

        None: Displays plot of vario per feature and a histogram of the values on different axes.
    Raises:
        ValueError: If variance_ratios is not a sequence of numbers.
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import pandas as pd
    from matplotlib.figure import Figure

    # ---- Validation ----

    if not isinstance(variance_ratios, np.ndarray):
        raise ValueError("variance_ratios must be a NumPy array.")
    if variance_ratios.ndim != 2:
        raise ValueError("variance_ratios must be a 2D array (num_pairs x num_features).")
    if not isinstance(pair_labels, list) or len(pair_labels) != variance_ratios.shape[0]:
        raise ValueError("pair_labels must be a list with the same length as the number of rows in variance_ratios.")
    
    figs = []
    for i, label in enumerate(pair_labels):
        fig = plt.figure(figsize=(14, 8))
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 8], wspace=0.3)

        # Histogram (left)
        ax1 = fig.add_subplot(gs[0])
        ax1.hist(variance_ratios[i], bins=20, orientation="horizontal", color=[0.8, 0.2, 0.2])
        ax1.set_xlabel("Frequency")
        ax1.invert_xaxis()
        ax1.yaxis.tick_right()
        ax1.yaxis.set_label_position("right")

        # Bar plot (right)
        ax2 = fig.add_subplot(gs[1], sharey=ax1)
        indices = np.arange(variance_ratios.shape[1])
        ax2.plot(indices, variance_ratios[i], "b-")
        ax2.plot(indices, variance_ratios[i], "r.")

        # Labels and title
        ax2.set_xlabel("Feature Index")
        ax2.set_ylabel("Variance Ratio: $(\\sigma_1 / \\sigma_2)$")
        ax2.set_title(f"Feature wise ratio of variance between {label}")
        ax2.grid(True)

        caption_i = caption or f"Variance ratio — {pair_labels[i]}"

        if rep is not None:
            rep.log_plot(fig, caption_i)
            plt.close(fig)
        else:
            figs.append((caption_i, fig))
            if show:
                plt.show()

    return None if rep is not None else figs

def levenes_plot(
    levenes_results: pd.DataFrame,
    *,
    rep = None,            # optional StatsReporter
    caption: Optional[str] = None,
    show: bool = False,
    pair_labels: Optional[list] = None
) -> plt.Figure:
    """
    Plots Levene's test results as a bar plot with significance markers.

    Args:
        levenes_results dict: pairlabels (batchwise), statistics, pvalues for each unique pair, statistic and pvalue are 1D arrays length of num_features"""
    figs = []
    return figs

"""----------------------------------------------------------------------------------------------------------------------------"""
"""---------------------------------------- Plotting functions for PCA correlation results ----------------------------------"""
"""----------------------------------------------------------------------------------------------------------------------------"""
@rep_plot_wrapper
def PC_corr_plot(
    PrincipleComponents,
    batch,
    covariates=None,
    variable_names=None,
    PC_correlations=False,
    *,
    show: bool = False,
    cluster_batches: bool = False
):
    """
    Generate multiple PCA diagnostic plots and return a list of (caption, fig).

    Improvements / behavior:
      - covariates may be a numpy array (2D), a pandas.DataFrame, or a structured numpy array.
      - If covariates has column names (DataFrame.columns or structured dtype.names), those names are used.
      - If covariates is a plain ndarray, variable_names (if provided) will be used as covariate names.
      - variable_names may optionally include 'batch' as the first element: ['batch', 'Age', 'Sex'].
      - If no covariate names are available, defaults "Covariate1", "Covariate2", ...


      K-means clustering of PCA points by batch and covariates to be added in future edit, additionally, 
      silhouette score calculation for batch also added. (Future work may add similar implementation for covariates if needed).

    Args:
        PrincipleComponents (np.ndarray): 2D array of PCA components (samples x components).
        batch (np.ndarray): 1D array of batch labels for each sample.
        covariates (Optional[Union[np.ndarray, pd.DataFrame]]): Optional covariate data.
        variable_names (Optional[List[str]]): Optional list of variable names for batch and covariates.
        PC_correlations (bool): If True, generate correlation heatmap.
        show (bool): If True, display plots immediately.
        
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    figs = []

    # Basic validation
    if not isinstance(PrincipleComponents, np.ndarray) or PrincipleComponents.ndim != 2:
        raise ValueError("PrincipleComponents must be a 2D numpy array (samples x components).")
    if not isinstance(batch, np.ndarray) or batch.ndim != 1:
        raise ValueError("batch must be a 1D numpy array.")
    if PrincipleComponents.shape[0] != len(batch):
        raise ValueError("Number of samples in PrincipleComponents and batch must match.")
    unique_batches = np.unique(batch)
    if len(unique_batches) < 2:
        raise ValueError("At least two unique batches are required.")

    # Build DataFrame of PCs
    PC_Names = [f"PC{i+1}" for i in range(PrincipleComponents.shape[1])]
    df = pd.DataFrame(PrincipleComponents, columns=PC_Names)

    # Decide batch column name (allow variable_names to include 'batch' as first element)
    batch_col_name = "batch"
    # If variable_names explicitly provided and starts with "batch", capture it as possible batch name
    if variable_names is not None and len(variable_names) > 0 and str(variable_names[0]).lower() == "batch":
        # use the exact provided first name (preserve case) as batch label
        batch_col_name = variable_names[0]
    

    df[batch_col_name] = batch
    # Change batch to numeric codes to prevent issues in plotting and calculating correlation:

    # --- Handle covariates robustly and determine covariate names ---
    cov_names = []
    cov_matrix = None  # numeric matrix (n_samples x n_covariates) used for correlations/plots

    if covariates is not None:
        # If DataFrame: use its column names
        if isinstance(covariates, pd.DataFrame):
            cov_matrix = covariates.values
            cov_names = list(map(str, covariates.columns))
        # Structured numpy array with named fields
        elif isinstance(covariates, np.ndarray) and covariates.dtype.names is not None:
            cov_names = [str(n) for n in covariates.dtype.names]
            # stack named columns into a 2D array
            cov_matrix = np.vstack([covariates[name] for name in cov_names]).T
        else:
            # array-like (convert to ndarray)
            cov_matrix = np.asarray(covariates)
            if cov_matrix.ndim != 2:
                raise ValueError("covariates must be 2D (samples x num_covariates).")
            if cov_matrix.shape[0] != PrincipleComponents.shape[0]:
                raise ValueError("Number of rows in covariates must match number of samples.")

            # If variable_names provided: it may either be exactly covariate names,
            # or include 'batch' as first element followed by covariate names.
            if variable_names is not None:
                # If user included 'batch' as first element, strip it.
                if len(variable_names) == cov_matrix.shape[1] + 1 and str(variable_names[0]).lower() == "batch":
                    cov_names = [str(x) for x in variable_names[1:]]
                elif len(variable_names) == cov_matrix.shape[1]:
                    cov_names = [str(x) for x in variable_names]
                else:
                    # inconsistent lengths: raise helpful error
                    raise ValueError(
                        "variable_names length does not match number of covariates.\n"
                        f"covariates has {cov_matrix.shape[1]} columns, "
                        f"but variable_names has length {len(variable_names)}.\n"
                        "If you include 'batch' in variable_names, put it first (e.g. ['batch', 'Age', 'Sex'])."
                    )
            else:
                # No variable_names: create defaults
                cov_names = [f"Covariate{i+1}" for i in range(cov_matrix.shape[1])]

        # Finally, assign covariate columns to df using cov_names
        # (if we reached here cov_matrix and cov_names should be set)
        if cov_matrix is None:
            raise ValueError("Unable to interpret covariates input; please supply a DataFrame, structured array, or 2D ndarray.")
        # Double-check shapes
        if cov_matrix.shape[0] != PrincipleComponents.shape[0]:
            raise ValueError("Number of rows in covariates must match number of samples.")
        if cov_matrix.shape[1] != len(cov_names):
            # defensive: if Pandas columns count mismatch (shouldn't happen), regenerate names
            cov_names = [f"Covariate{i+1}" for i in range(cov_matrix.shape[1])]

        for i, name in enumerate(cov_names):
            df[name] = cov_matrix[:, i]
    else:
        # No covariates present; ensure variable_names is either None or only contains 'batch'
        if variable_names is not None:
            if not (len(variable_names) == 1 and str(variable_names[0]).lower() == "batch"):
                raise ValueError("variable_names provided but covariates is None. Provide covariates or remove variable_names.")
        cov_names = []

    # --- 1) PCA scatter by batch ---
    fig1, ax = plt.subplots(figsize=(8, 6))
    for b in unique_batches:
        ax.scatter(df.loc[df[batch_col_name] == b, "PC1"], df.loc[df[batch_col_name] == b, "PC2"], label=f"{batch_col_name} {b}", alpha=0.7)
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_title("PCA Scatter Plot by Batch")
    ax.legend()
    ax.grid(True)
    figs.append(("PCA scatter by batch", fig1))
    
    batch_numeric = pd.Categorical(batch).codes
    batch_col_code = f"{batch_col_name}_code"
    df[batch_col_code] = batch_numeric
    # --- 2) PCA scatter by each covariate (if present) ---
    if cov_names:
        for name in cov_names:
            vals = df[name].values
            fig, ax = plt.subplots(figsize=(8, 6))
            # treat small-unique-count as categorical
            if len(np.unique(vals)) <= 20:
                for cat in np.unique(vals):
                    sel = df[name] == cat
                    ax.scatter(df.loc[sel, "PC1"], df.loc[sel, "PC2"], label=f"{name}={cat}", alpha=0.6)
            else:
                sc = ax.scatter(df["PC1"], df["PC2"], c=vals, cmap="viridis", alpha=0.7)
                plt.colorbar(sc, ax=ax, label=name)
            ax.set_xlabel("Principal Component 1")
            ax.set_ylabel("Principal Component 2")
            ax.set_title(f"PCA Scatter Plot by {name}")
            # legend can be large; show only for categorical
            if len(np.unique(vals)) <= 20:
                ax.legend(loc="best", fontsize="small", frameon=True)
            ax.grid(True)
            figs.append((f"PCA scatter by {name}", fig))

    # --- 3) Correlation heatmap if requested ---
    if PC_correlations:
        # create combined_data and combined_names in the same order used for corr matrix
        if cov_names:
            combined_data = np.column_stack((PrincipleComponents, df[batch_col_code].values.reshape(-1, 1), df[cov_names].values))
            combined_names = PC_Names + [batch_col_code] + cov_names
        else:
            combined_data = np.column_stack((PrincipleComponents, df[batch_col_code].values.reshape(-1, 1)))
            combined_names = PC_Names + [batch_col_code]

        corr = np.corrcoef(combined_data.T)
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", xticklabels=combined_names, yticklabels=combined_names, ax=ax)
        ax.set_title("Correlation Matrix of PCs, Batch, and Covariates")
        figs.append(("PCA correlation matrix", fig))
    
    # show only if requested
    if show:
        for _, f in figs:
            try:
                f.show()
            except Exception:
                # some backends may not support show on Figure objects; ignore safely
                pass

    return figs
"""----------------------------------------------------------------------------------------------------------------------------"""
"""---------------------------------------- Plotting functions for PCA clustering results ----------------------------------"""
"""----------------------------------------------------------------------------------------------------------------------------"""
@rep_plot_wrapper # This is redundant now so may remove at later date
def pc_clustering_plot(
    PrincipleComponents,
    batch,
    covariates=None,
    variable_names=None,
    n_pcs_for_clustering=None,
    n_clusters_for_kmeans=None,
    random_state=0,
    *,
    show=False
):
    """
    NOTE TO USER: THIS FUNCTION IS A NEW ADDITION AND WAS PARTIALLY CREATED USING CHATGPT. PLEASE REVIEW CAREFULLY.
    Compute clustering diagnostics on PCA (or any embedding).

    Inputs:
      - PrincipleComponents: ndarray (n_samples x n_components)
      - batch: 1D array-like labels (length = n_samples)
      - covariates: optional (not used for clustering but kept for API parity)
      - variable_names: optional list (keeps same semantics as your other function)
      - n_pcs_for_clustering: int or None (default = min(10, n_components))
      - n_clusters_for_kmeans: int or None (default = number of unique batches)
      - random_state: int
      - show: bool -> call fig.show() if True

    Returns:
      - figs: list of (caption, matplotlib.Figure)
      - metrics: dict with silhouette, ARI, NMI, contingency table, chi2, km_labels, etc.
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    from sklearn.preprocessing import LabelEncoder
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
    from scipy.stats import chi2_contingency

    # --- input validation & normalization (mirrors your style) ---
    if not isinstance(PrincipleComponents, np.ndarray) or PrincipleComponents.ndim != 2:
        raise ValueError("PrincipleComponents must be a 2D numpy array (samples x components).")
    if not isinstance(batch, np.ndarray):
        batch = np.asarray(batch)
    if batch.ndim != 1:
        raise ValueError("batch must be a 1D array-like.")
    if PrincipleComponents.shape[0] != batch.shape[0]:
        raise ValueError("Number of samples in PrincipleComponents and batch must match.")
    n_samples, n_components = PrincipleComponents.shape
    unique_batches = np.unique(batch)
    if len(unique_batches) < 1:
        raise ValueError("batch must contain at least one label.")

    # choose number of PCs to use for clustering diagnostics
    if n_pcs_for_clustering is None:
        n_pcs_for_clustering = min(10, n_components)
    else:
        n_pcs_for_clustering = min(int(n_pcs_for_clustering), n_components)
    X = PrincipleComponents[:, :n_pcs_for_clustering]

    # determine k for KMeans
    n_batches = len(unique_batches)
    k = n_clusters_for_kmeans or n_batches
    k = int(k)
    if not (1 <= k <= n_samples):
        raise ValueError("n_clusters_for_kmeans must be between 1 and n_samples")

    # label encode batch for metric functions
    le = LabelEncoder()
    try:
        batch_enc = le.fit_transform(batch)
    except Exception:
        batch_enc = le.fit_transform(batch.astype(str))

    figs = []
    metrics = {}

    # --- silhouette using batch as labels (if valid) ---
    if 2 <= n_batches <= (n_samples - 1):
        try:
            sil = silhouette_score(X, batch_enc)
            metrics["silhouette_using_batch"] = float(sil)
        except Exception as e:
            metrics["silhouette_using_batch"] = None
            metrics["silhouette_error"] = str(e)
    else:
        metrics["silhouette_using_batch"] = None
        metrics["silhouette_note"] = "silhouette requires 2 <= n_labels <= n_samples-1"

    # --- KMeans clustering ---
    # handle sklearn's n_init compatibility ('auto' introduced in newer sklearn)
    try:
        km = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
    except TypeError:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    km_labels = km.fit_predict(X)
    metrics["kmeans_n_clusters"] = int(k)
    metrics["kmeans_labels"] = km_labels

    # ARI / NMI against batch
    try:
        ari = adjusted_rand_score(batch_enc, km_labels)
        nmi = normalized_mutual_info_score(batch_enc, km_labels)
        metrics["kmeans_ari_vs_batch"] = float(ari)
        metrics["kmeans_nmi_vs_batch"] = float(nmi)
    except Exception as e:
        metrics["kmeans_ari_vs_batch"] = None
        metrics["kmeans_nmi_vs_batch"] = None
        metrics["kmeans_metrics_error"] = str(e)

    # contingency table + chi-square test
    ct = pd.crosstab(pd.Series(batch, name="batch"), pd.Series(km_labels, name="kmeans_label"))
    metrics["contingency_table_batch_vs_kmeans"] = ct
    try:
        chi2, pval, dof, expected = chi2_contingency(ct)
        metrics["chi2_vs_kmeans"] = {"chi2": float(chi2), "pvalue": float(pval), "dof": int(dof)}
    except Exception as e:
        metrics["chi2_vs_kmeans"] = {"error": str(e)}

    # --- Figures: KMeans scatter, compare vs batch, silhouette per-batch plot (if silhouette computed) ---
    # 1) KMeans clusters (PC1 vs PC2)
    fig_km, ax = plt.subplots(figsize=(8, 6))
    for lbl in np.unique(km_labels):
        sel = km_labels == lbl
        ax.scatter(X[sel, 0], X[sel, 1], label=f"k={lbl}", alpha=0.7, s=35)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"KMeans (k={k}) on first {n_pcs_for_clustering} PCs")
    ax.legend(loc="best", fontsize="small")
    ax.grid(True)
    figs.append((f"KMeans clustering (k={k})", fig_km))

    # 2) Side-by-side comparison: colored by batch vs colored by kmeans
    fig_cmp, axs = plt.subplots(1, 2, figsize=(14, 5))
    # by batch
    for b in unique_batches:
        sel = batch == b
        axs[0].scatter(X[sel, 0], X[sel, 1], label=str(b), alpha=0.6, s=30)
    axs[0].set_title("By batch")
    axs[0].set_xlabel("PC1"); axs[0].set_ylabel("PC2"); axs[0].legend(fontsize="small")
    # by kmeans
    for lbl in np.unique(km_labels):
        sel = km_labels == lbl
        axs[1].scatter(X[sel, 0], X[sel, 1], label=f"k={lbl}", alpha=0.6, s=30)
    axs[1].set_title("By KMeans cluster")
    axs[1].set_xlabel("PC1"); axs[1].set_ylabel("PC2"); axs[1].legend(fontsize="small")
    fig_cmp.suptitle("Compare batch vs kmeans (PC1 vs PC2)")
    figs.append(("Compare batch vs kmeans", fig_cmp))

    # 3) Optional: silhouette per batch (if silhouette computed)
    if metrics.get("silhouette_using_batch") is not None:
        # compute individual sample silhouettes and average per batch
        try:
            from sklearn.metrics import silhouette_samples
            sample_sil = silhouette_samples(X, batch_enc)
            sil_by_batch = {}
            for b in unique_batches:
                sel = (batch == b)
                if sel.sum() > 0:
                    sil_by_batch[str(b)] = float(np.nanmean(sample_sil[sel]))
                else:
                    sil_by_batch[str(b)] = None
            metrics["silhouette_by_batch"] = sil_by_batch

            # plotting
            fig_sil, ax = plt.subplots(figsize=(8, 4))
            names = list(sil_by_batch.keys())
            vals = [sil_by_batch[n] if sil_by_batch[n] is not None else np.nan for n in names]
            ax.bar(names, vals)
            ax.set_ylabel("Average silhouette (per-batch)")
            ax.set_title("Average silhouette score per batch")
            figs.append(("Silhouette per batch", fig_sil))
        except Exception as e:
            metrics["silhouette_by_batch_error"] = str(e)
    # show if requested
    if show:
        for _, f in figs:
            try:
                f.show()
            except Exception:
                pass

    return figs

"""----------------------------------------------------------------------------------------------------------------------------"""
"""Plotting per batch variance across first N PCs"""
"""----------------------------------------------------------------------------------------------------------------------------"""
def plot_eigen_spectra_and_cumulative(
    score: np.ndarray,
    batch: np.ndarray,
    rep,
    max_components: int = 50,
    caption_prefix: str = "PC spectrum",
) -> dict[str, Any]:
    """
    Compute per-batch variance along PCs (scree / cumulative) and log plots to the report.

    Args:
        score: (n_samples, n_pcs) PCA score matrix returned by your PCA routine.
        batch: (n_samples,) batch labels (numeric or strings).
        rep: report object that has rep.log_plot(plt, caption=...) and rep.log_text(...)
        max_components: maximum number of PCs to visualise (keeps plots cheap).
        caption_prefix: prefix for plot captions.

    Returns:
        results dict containing:
          - 'per_batch_variance': {batch_label: 1D-array length k of variances per PC}
          - 'per_batch_frac_var': {batch_label: 1D-array length k of fraction variance (per-batch)}
          - 'pcs_used': k
    """
    # Basic checks
    if score.ndim != 2:
        raise ValueError("score must be a 2D array (n_samples x n_pcs)")

    n, n_pcs = score.shape
    k = min(n_pcs, max_components)
    if k < 2:
        rep.log_text("Not enough PCs to produce spectrum plots (k < 2).")
        return {}

    unique_batches = np.unique(batch)
    per_batch_variance = {}
    per_batch_frac_var = {}

    # Compute per-batch variance along each PC (diagonal / axis variances)
    for b in unique_batches:
        idx = np.where(batch == b)[0]
        if len(idx) < 2:
            # Variance undefined or uninformative
            rep.log_text(f"Batch {b}: too few samples ({len(idx)}) to estimate per-PC variance reliably.")
            per_batch_variance[b] = np.full(k, np.nan)
            per_batch_frac_var[b] = np.full(k, np.nan)
            continue
        scores_b = score[idx, :k]
        # ddof=1 to mirror sample covariance convention
        var_b = np.nanvar(scores_b, axis=0, ddof=1)
        total_var_b = np.nansum(var_b)
        if total_var_b == 0 or np.isnan(total_var_b):
            frac = np.full_like(var_b, np.nan)
        else:
            frac = var_b / total_var_b
        per_batch_variance[b] = var_b
        per_batch_frac_var[b] = frac

    results = {
        "pcs_used": k,
        "per_batch_variance": per_batch_variance,
        "per_batch_frac_var": per_batch_frac_var,
    }

        # --- One figure with two horizontally aligned subplots ---
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14, 4.5),
        sharex=True
    )

    # --- Scree plot (fraction of variance per PC) ---
    for b in unique_batches:
        frac = per_batch_frac_var[b]
        if np.all(np.isnan(frac)):
            continue
        ax1.plot(
            np.arange(1, k + 1),
            frac,
            marker="o",
            label=f"Batch {b}",
            alpha=0.8
        )

    ax1.set_xlabel("PC index")
    ax1.set_ylabel("Fraction variance (per-batch)")
    ax1.set_title("Per-batch scree plot")
    ax1.grid(axis="y", alpha=0.2)
    ax1.legend(frameon=False, fontsize="small")

    # --- Cumulative variance plot (per-batch) ---
    for b in unique_batches:
        frac = per_batch_frac_var[b]
        if np.all(np.isnan(frac)):
            continue
        cum = np.nancumsum(frac)
        ax2.plot(
            np.arange(1, k + 1),
            cum,
            marker="o",
            label=f"Batch {b}",
            alpha=0.8
        )

    ax2.set_xlabel("PC index")
    ax2.set_ylabel("Cumulative fraction variance")
    ax2.set_title("Per-batch cumulative variance explained")
    ax2.set_ylim(0, 1.05)
    ax2.grid(axis="y", alpha=0.2)

    plt.tight_layout()
    rep.log_plot(
        plt,
        caption=f"{caption_prefix}: Per-batch scree + cumulative variance explained"
    )
    plt.close()

    # short textual summary (helpful for users)
    rep.log_text(
        f"{caption_prefix}: Used first {k} PCs. "
        "Scree and cumulative plots show how variance is distributed across PCs per batch."
    )

    return results
"""----------------------------------------------------------------------------------------------------------------------------"""
"""---------------------------------------- Plotting covariance Frobenius norm results ----------------------------------"""
"""----------------------------------------------------------------------------------------------------------------------------"""

def plot_covariance_frobenius(
    score: np.ndarray,
    batch: np.ndarray,
    rep,
    max_components: int = 50,
    normalize: bool = True,
    caption_prefix: str = "Covariance comparison (PC space)",
) -> dict[str, Any]:
    """
    Compute pairwise Frobenius norms of covariance differences between batches using PC scores.

    Args:
        score: (n_samples, n_pcs) PCA score matrix.
        batch: (n_samples,) batch labels.
        rep: report object (must support rep.log_plot and rep.log_text).
        max_components: number of PCs (k) to use for covariance estimation.
        normalize: if True, divide pairwise norms by Frobenius norm of pooled covariance.
        caption_prefix: prefix for plot captions.

    Returns:
        results dict containing:
          - 'cov_matrices': {batch_label: k x k covariance matrix}
          - 'pairwise_frobenius': DataFrame-like dict or 2D np.array of pairwise norms
          - 'pairwise_frobenius_normalized': same but normalized (if normalize=True)
          - 'pcs_used': k
    """
    import pandas as pd  

    if score.ndim != 2:
        raise ValueError("score must be a 2D array (n_samples x n_pcs)")

    n, n_pcs = score.shape
    k = min(n_pcs, max_components)
    if k < 1:
        rep.log_text("Not enough PCs to compute covariance diagnostics.")
        return {}

    unique_batches = np.unique(batch)
    G = len(unique_batches)

    cov_matrices = {}
    valid_batches = []
    for b in unique_batches:
        idx = np.where(batch == b)[0]
        if len(idx) < 2:
            rep.log_text(f"Batch {b}: too few samples ({len(idx)}) to estimate covariance reliably.")
            # store NaN matrix to preserve indexing
            cov_matrices[b] = np.full((k, k), np.nan)
            continue
        scores_b = score[idx, :k]
        cov_b = np.cov(scores_b, rowvar=False, ddof=1)  # k x k
        cov_matrices[b] = cov_b
        valid_batches.append(b)

    # pooled covariance (using all samples)
    pooled_scores = score[:, :k]
    pooled_cov = np.cov(pooled_scores, rowvar=False, ddof=1)
    pooled_frob = np.linalg.norm(pooled_cov, ord='fro')

    # pairwise frobenius norms
    pairwise = np.full((G, G), np.nan)
    batch_list = list(unique_batches)
    for i, bi in enumerate(batch_list):
        for j, bj in enumerate(batch_list):
            Ci = cov_matrices[bi]
            Cj = cov_matrices[bj]
            if np.isnan(Ci).all() or np.isnan(Cj).all():
                pairwise[i, j] = np.nan
            else:
                diff = Ci - Cj
                pairwise[i, j] = np.linalg.norm(diff, ord='fro')

    # normalized version
    pairwise_norm = pairwise.copy()
    if normalize and pooled_frob > 0 and not np.isnan(pooled_frob):
        pairwise_norm = pairwise / pooled_frob

    # Turn results into a pandas DataFrame for nicer human-readable output if desired
    try:
        df_pairwise = pd.DataFrame(pairwise, index=batch_list, columns=batch_list)
        df_pairwise_norm = pd.DataFrame(pairwise_norm, index=batch_list, columns=batch_list)
    except Exception:
        df_pairwise = pairwise
        df_pairwise_norm = pairwise_norm

    results = {
        "pcs_used": k,
        "cov_matrices": cov_matrices,
        "pairwise_frobenius": df_pairwise,
        "pairwise_frobenius_normalized": df_pairwise_norm,
        "pooled_frobenius": pooled_frob,
    }
        # --- One figure with two horizontally aligned subplots ---
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14, 5),
        gridspec_kw={"width_ratios": [3, 2]}
    )

    # --- Heatmap of normalized pairwise differences ---
    im = ax1.imshow(pairwise_norm, interpolation="nearest", aspect="auto")

    # Add value to each cell
    for i in range(G):
        for j in range(G):
            val = pairwise_norm[i, j]
            if not np.isnan(val):
                ax1.text(j, i, f"{val:.2f}", ha="center", va="center", color="black")

    cbar = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    ax1.set_xticks(np.arange(G))
    ax1.set_xticklabels(batch_list, rotation=45, ha="right")
    ax1.set_yticks(np.arange(G))
    ax1.set_yticklabels(batch_list)
    ax1.set_title("Normalized Frobenius norm of covariance differences (PC space)")

    # --- Bar plot: max per batch ---
    max_per_batch = np.nanmax(pairwise_norm, axis=1)
    ax2.bar(range(G), max_per_batch)
    ax2.set_xticks(range(G))
    ax2.set_xticklabels(batch_list, rotation=45, ha="right")
    ax2.set_ylabel("Max normalized Frobenius distance")
    ax2.set_title("Max covariance difference per batch (normalized)")

    plt.tight_layout()

    rep.log_plot(plt, caption=f"{caption_prefix}: Pairwise normalized Frobenius norms (heatmap)")
    plt.close()

    # --- Bar plot showing max difference per batch (summary) ---
    # compute max distance from each batch to others

    rep.log_text(
        f"{caption_prefix}: used first {k} PCs. Pooled Frobenius norm = {pooled_frob:.4g}. "
        "Pairwise normalized Frobenius matrix and per-batch summaries added to report."
    )
    return results

"""----------------------------------------------------------------------------------------------------------------------------"""
"""---------------------------------------- Plotting functions for Mahalanobis distance ----------------------------------"""
"""----------------------------------------------------------------------------------------------------------------------------"""
def mahalanobis_distance_plot(results: dict,
                               rep=None,
                                 annotate: bool = True,
                                   figsize=(14,5),
                                     cmap="viridis",
                                       show: bool = False):

    """
    Plot Mahalanobis distances from (...) all on ONE figure:
      - Heatmap of pairwise RAW distances
      - Heatmap of pairwise RESIDUAL distances (if available)
      - Bar chart of centroid-to-global distances (raw vs residual)

    Args:
        results (dict): Output from MahalanobisDistance(...)
        annotate (bool): Write numeric values inside heatmap cells/bars.
        figsize (tuple): Matplotlib figure size.
        cmap (str): Colormap for heatmaps.
        show (bool): If True, plt.show(); otherwise just return (fig, axes).

    Returns:
        (fig, axes): The matplotlib Figure and dict of axes.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    # ---- Validation ----
    if not isinstance(results, dict):
        raise ValueError("results must be a dict produced by MahalanobisDistance(...)")

    req = ["pairwise_raw", "centroid_raw", "batches"]
    for k in req:
        if k not in results:
            raise ValueError(f"Missing required key '{k}' in results.")
    # Optional
    pairwise_resid = results.get("pairwise_resid", None)
    centroid_resid = results.get("centroid_resid", None)

    pairwise_raw = results["pairwise_raw"]
    centroid_raw = results["centroid_raw"]
    batches = results["batches"]
    if isinstance(batches, np.ndarray):
        batches = batches.tolist()
    n = len(batches)
    if n < 2:
        raise ValueError("Need at least two batches to plot distances.")

    # ---- Helpers ----
    def build_matrix(pw: dict) -> np.ndarray:
        M = np.full((n, n), np.nan, dtype=float)
        # Fill symmetric entries from pairwise dict keys (b1, b2)
        # Diagonal defined as 0 (distance of a batch to itself)
        for i in range(n):
            M[i, i] = 0.0
        if pw is None:
            return M
        for (b1, b2), d in pw.items():
            i = batches.index(b1)
            j = batches.index(b2)
            M[i, j] = d
            M[j, i] = d
        return M

    def centroid_array(cent: dict) -> np.ndarray:
        if cent is None:
            return None
        # keys like (b, 'global')
        return np.array([float(cent[(b, "global")]) for b in batches], dtype=float)

    def annotate_heatmap(ax, M):
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8)

    # ---- Data prep ----
    M_raw = build_matrix(pairwise_raw)
    M_resid = build_matrix(pairwise_resid) if pairwise_resid is not None else None

    # Use a shared color scale across heatmaps for fair comparison
    vmax_candidates = [np.nanmax(M_raw)]
    if M_resid is not None:
        vmax_candidates.append(np.nanmax(M_resid))
    vmax = np.nanmax(vmax_candidates)
    vmin = 0.0

    c_raw = centroid_array(centroid_raw)
    c_res = centroid_array(centroid_resid) if centroid_resid is not None else None

    # ---- Figure layout ----
    # If residuals exist: 3 panels (raw, resid, bars)
    # Else: 2 panels (raw, bars)
    has_resid = (pairwise_resid is not None) and (centroid_resid is not None)
    num_cols = 3 if has_resid else 2

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(1, num_cols, figure=fig, width_ratios=[1, 1, 0.9] if has_resid else [1, 1])

    ax_raw = fig.add_subplot(gs[0, 0])
    im_raw = ax_raw.imshow(M_raw, cmap=cmap, vmin=vmin, vmax=vmax)
    ax_raw.set_title("Pairwise Mahalanobis (Raw)")
    ax_raw.set_xticks(range(n))
    ax_raw.set_yticks(range(n))
    ax_raw.set_xticklabels(batches, rotation=45, ha="right")
    ax_raw.set_yticklabels(batches)
    ax_raw.set_xlabel("Batch")
    ax_raw.set_ylabel("Batch")
    if annotate:
        annotate_heatmap(ax_raw,M_raw)

    if has_resid:
        ax_resid = fig.add_subplot(gs[0, 1])
        im_resid = ax_resid.imshow(M_resid, cmap=cmap, vmin=vmin, vmax=vmax)
        ax_resid.set_title("Pairwise Mahalanobis (Residual)")
        ax_resid.set_xticks(range(n))
        ax_resid.set_yticks(range(n))
        ax_resid.set_xticklabels(batches, rotation=45, ha="right")
        ax_resid.set_yticklabels(batches)
        ax_resid.set_xlabel("Batch")
        ax_resid.set_ylabel("Batch")
        if annotate:
            annotate_heatmap(ax_resid,M_resid)

        # One colorbar shared by both heatmaps
        cbar = fig.colorbar(im_resid, ax=ax_raw, fraction=0.046, pad=0.2,orientation="horizontal",location="top")
        cbar = fig.colorbar(im_resid, ax=ax_resid, fraction=0.046, pad=0.2,orientation="horizontal",location="top")

        cbar.set_label("Mahalanobis distance")
    else:
        # Single colorbar for the single heatmap
        cbar = fig.colorbar(im_raw, ax=ax_raw, fraction=0.046, pad=0.04)
        cbar.set_label("Mahalanobis distance")

    # ---- Bar chart of centroid-to-global ----
    ax_bar = fig.add_subplot(gs[0, -1])
    x = np.arange(n)
    if c_res is None:
        # Only raw bars
        width = 0.6
        bars = ax_bar.bar(x, c_raw, width, label="Raw")
        ax_bar.set_title("Centroid → Global")
        if annotate:
            for b in bars:
                ax_bar.text(b.get_x() + b.get_width()/2., b.get_height(),
                            f"{b.get_height():.2f}",
                            ha='center', va='bottom', fontsize=8)
        ax_bar.legend()
    else:
        width = 0.38
        bars_raw = ax_bar.bar(x - width/2, c_raw, width, label="Raw")
        bars_res = ax_bar.bar(x + width/2, c_res, width, label="Residual")
        ax_bar.set_title("Centroid → Global (Raw vs Residual)")
        if annotate:
            for b in list(bars_raw) + list(bars_res):
                ax_bar.text(b.get_x() + b.get_width()/2., b.get_height(),
                            f"{b.get_height():.2f}",
                            ha='center', va='bottom', fontsize=8)
        ax_bar.legend()

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(batches, rotation=45, ha="right")
    ax_bar.set_ylabel("Mahalanobis distance")
    ax_bar.set_xlabel("Batch")

    axes = {"heatmap_raw": ax_raw, "bars": ax_bar}
    if has_resid:
        axes["heatmap_resid"] = ax_resid
    #fig.tight_layout()
    if rep is not None:
        rep.log_plot(fig, "Mahalanobis distances (raw vs residual)")
        plt.close(fig)
        return None, None  # or return a small marker that it was logged
    if show:
        plt.show()
    return fig, axes
"""----------------------------------------------------------------------------------------------------------------------------"""
"""---------------------------------------- Plotting functions for Mixed effects model ----------------------------------"""
"""----------------------------------------------------------------------------------------------------------------------------"""
# @rep_plot_wrapper To be added at a later date (currently plotted in the report directly as we haven't decided on a standard plot format)
"""----------------------------------------------------------------------------------------------------------------------------"""
"""---------------------------------------- Plotting functions for Two-sample Kolmogorov-Smirnov test ----------------------------------"""
"""----------------------------------------------------------------------------------------------------------------------------"""
def KS_plot(ks_results: dict,
             feature_names: list = None,
               rep = None,            # optional StatsReporter
                 caption: Optional[str] = None,
                   show: bool = False) -> plt.Figure:
    """
    Plot the output of the two sample KS test as ordered plots of the -log10 p-values for each feature.

    Overall, returns two plots
        - one plot showing the pairwise KS test results for each feature as a dot plot (ordered as -log10 p-value)
        - One plot showing the batch vs whole dataset (excluding that batch), again as a dot plot ordered by -log10 p-value.

    Args:
        ks_results (dict): Output from TwoSampleKSTest(...)
            ks_results: keys are tuples like (b, 'overall') or (b1, b2)
        - each value is a dict with:
            'statistic': np.array of D statistics (length n_features)
            'p_value': np.array of p-values (nan where test not run)
            'p_value_fdr': np.array of BH-corrected p-values (if do_fdr else None)
            'n_group1': array of sample counts per feature for group1 (same across features but kept for completeness)
            'n_group2': array of counts for group2
            'summary': {'prop_significant': float, 'mean_D': float}
    Returns:
        figs (list): List of (caption, fig) tuples for each plot generated.

    """
    # ---- Validation ---- Structure of dictionary has batch vs over all and batch vs batch as keys

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.figure import Figure
    from matplotlib.pyplot import gca    
# plot batch vs overall on one plot and code with the legend:
    fig= plt.figure(figsize=(12, 6))
    figs = []
    ax = gca()

    for key in ks_results:
        if len(key) == 2 and key[1] == 'overall':
            b = key[0]
            res = ks_results[key]
            p_values = res['p_value']
            if feature_names is not None and len(feature_names) != len(p_values):
                raise ValueError("feature_names length must match number of features in ks_results.")
            n_features = len(p_values)
            indices = np.arange(n_features)
            # sort by -log10 p-value
            sorted_indices = np.argsort(-np.log10(p_values + 1e-10))  # add small value to avoid log(0)
            sorted_pvals = p_values[sorted_indices]
            sorted_features = feature_names[sorted_indices] if feature_names is not None else sorted_indices
            ax.plot(indices, -np.log10(sorted_pvals + 1e-10), '*',label=f'Batch {b} vs Overall')
    plt.xlabel("Features (ordered by -log10 p-value)")
    plt.ylabel("-log10 p-value")
    plt.title("KS Test: Batch vs Overall")
    plt.grid(True)
    plt.legend()
    sig_threshold_05 = -np.log10(0.05 / n_features)
    sig_threshold_01 = -np.log10(0.01 / n_features)
    plt.axhline(y=sig_threshold_05, color='r', linestyle='-', label='Significance Threshold (0.05 Bonferroni)')
    plt.axhline(y=sig_threshold_01, color='g', linestyle='-', label='Significance Threshold (0.01 Bonferroni)')
    figs.append((caption or "KS Test: Batch vs Overall", fig))

    # Repeat for batch vs batch on next figure:
    fig2 = plt.figure(figsize=(12, 6))
    ax2 = gca()
    for key in ks_results:
        if len(key) == 2 and key[1] != 'overall':
            b = key[0]
            res = ks_results[key]
            p_values = res['p_value']
            if feature_names is not None and len(feature_names) != len(p_values):
                raise ValueError("feature_names length must match number of features in ks_results.")
            n_features = len(p_values)
            indices = np.arange(n_features)
            # sort by -log10 p-value
            sorted_indices = np.argsort(-np.log10(p_values + 1e-10))  # add small value to avoid log(0)
            sorted_pvals = p_values[sorted_indices]
            sorted_features = feature_names[sorted_indices] if feature_names is not None else sorted_indices
            ax2.plot(indices, -np.log10(sorted_pvals + 1e-10),'.', label=f'Batch {b} vs Overall')
    plt.xlabel("Features (ordered by -log10 p-value)")
    plt.ylabel("-log10 p-value")
    plt.title("KS Test: Batch vs Batch")
    plt.grid(True)
    # Add an line to the plot to indicate significant threshold at 0.05 and 0.01 (Bonferroni corrected and uncorrected)
    sig_threshold_05 = -np.log10(0.05 / n_features)
    sig_threshold_01 = -np.log10(0.01 / n_features)
    plt.axhline(y=sig_threshold_05, color='r', linestyle='-', label='Significance Threshold (0.05 Bonferroni)')
    plt.axhline(y=sig_threshold_01, color='g', linestyle='-', label='Significance Threshold (0.01 Bonferroni)')

    plt.legend()
    figs.append((caption or "KS Test: Batch vs Batch", fig2))

    # Check if show is given, if so, display the plots
    for caption_i, fig in figs:
        if rep is not None:
            rep.log_plot(fig, caption_i)
            plt.close(fig)
        else: figs.append((caption_i, fig))
    if show:
        for _, fig in figs:
            fig.show()
    return rep if rep is not None else figs

##########################################################################
# Plotting for longitudinal metrics
###########################################################################
import os, re, json, argparse
from typing import List, Dict, Tuple, Any
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from typing import Sequence, Optional
from matplotlib.figure import Figure

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})
STAR = u"\u2605"
_MARKERS = ["o","s","D","v","^","P","X","H","<",">","8","p","*"]

INTERP_FS_DELTA = 2

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def tidy_axes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    # NEW — add grids
    ax.grid(True, which='major', axis='both', linestyle='--', linewidth=0.6, color='#dddddd')
    ax.set_axisbelow(True)


def normalize_key(s: Any) -> str:
    if s is None: return ""
    s = str(s).strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^0-9a-z_]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")

def _tone_hex(hexcolor: str, factor: float = 0.88) -> str:
    hexcolor = hexcolor.lstrip("#")
    r = int(hexcolor[0:2], 16); g = int(hexcolor[2:4], 16); b = int(hexcolor[4:6], 16)
    r = max(0, min(255, int(r * factor)))
    g = max(0, min(255, int(g * factor)))
    b = max(0, min(255, int(b * factor)))
    return f"#{r:02x}{g:02x}{b:02x}"

class StyleBank:
    """Deterministic style assignment for keys (idp/site/timepair)."""
    def __init__(self):
        self._map = {}
        self._colors = plt.get_cmap("Accent").colors   # brighter palette
        self._next = 0
    def get(self, key: str) -> Dict[str,str]:
        nk = normalize_key(key)
        if nk in self._map:
            return self._map[nk]
        i = self._next
        color = mpl.colors.to_hex(self._colors[i % len(self._colors)])  # no darkening
        color = _tone_hex(color, 0.86)
        marker = _MARKERS[i % len(_MARKERS)]
        self._map[nk] = {"color": color, "marker": marker, "label": key}
        self._next += 1
        return self._map[nk]
    def items(self):
        return list(self._map.items())

def _style_fetch(stylebank, key, default):
    """
    Safe access for stylebank entries. Handles:
      - dict-like .get(k, default)
      - custom .get(k) (no default) returning None if missing
      - __getitem__ access stylebank[k]
    Always returns either the found value or `default`.
    """
    # If None provided, just return default
    if stylebank is None:
        return default
    # If it's an actual dict, use its get with default
    if isinstance(stylebank, dict):
        return stylebank.get(key, default)
    # Try calling get(key, default) (works for dict-like)
    try:
        return stylebank.get(key, default)
    except TypeError:
        # get exists but does not accept default: try get(key)
        try:
            val = stylebank.get(key)
            return default if val is None else val
        except Exception:
            pass
    except AttributeError:
        # no get method; try item access
        pass

    # Try dictionary-style access
    try:
        return stylebank[key]
    except Exception:
        # give up: return default
        return default

@rep_plot_wrapper  # uses the wrapper 
def plot_pairwise_spearman_combined(all_results, outdir) -> List[Tuple[str, Figure]]:
    """
    Build two summary plots and return them as [(label, Figure), ...].
    The wrapper will log & close figs if called with rep=... or log_func=...
    """
    stylebank = StyleBank()
    labels           = [lab for lab, _ in all_results]
    per_dir_idp_avgs = []; per_dir_tp_avgs = []; idps = []; tps = []

    for lab, dct in all_results:
        df = dct.get("pairwise_spearman", pd.DataFrame())
        if df is None or df.empty:
            per_dir_idp_avgs.append({}); per_dir_tp_avgs.append({}); continue
        if not set(['TimeA','TimeB','IDP','SpearmanRho']).issubset(df.columns):
            per_dir_idp_avgs.append({}); per_dir_tp_avgs.append({}); continue
        idp_avg = df.groupby('IDP')['SpearmanRho'].mean().to_dict()
        per_dir_idp_avgs.append(idp_avg); idps.extend(list(idp_avg.keys()))
        tp = df.groupby(['TimeA','TimeB'])['SpearmanRho'].mean()
        tp_idx = ['%s|%s' % (a,b) for (a,b) in tp.index.tolist()]
        per_dir_tp_avgs.append(dict(zip(tp_idx, tp.values.tolist()))); tps.extend(tp_idx)
    
    idps = list(dict.fromkeys(idps)); tps = list(dict.fromkeys(tps))

    figs: List[Tuple[str, Figure]] = []

    # -------------------------
    # IDP average boxplot + scatter
    # -------------------------
    boxdata = [[per_dir_idp_avgs[i].get(idp, np.nan) for idp in idps] for i in range(len(per_dir_idp_avgs))]
    fig1, ax1 = plt.subplots(figsize=(max(8,1.2*len(labels)),6)); tidy_axes(ax1)
    ax1.boxplot([ [v for v in arr if not np.isnan(v)] for arr in boxdata],
                positions=np.arange(1,len(labels)+1), widths=0.6, patch_artist=True,
                boxprops=dict(facecolor='none'))
        # scatter each idp
    for i, d in enumerate(per_dir_idp_avgs):
        for idp in idps:
            val = d.get(idp, np.nan)
            if np.isnan(val): 
                continue
            st = _style_fetch(stylebank, idp, {'marker':'o','color':'#1f77b4'})
            ax1.scatter(i+1 + np.random.uniform(-0.08,0.08), val,
                        marker=st.get('marker','o'), color=st.get('color','#1f77b4'),
                        edgecolor=_tone_hex(st.get('color','#1f77b4'),0.6), s=60)

    # IDP legend (build handles safely)
    if len(idps) > 0 and len(idps) <= 40:
        handles = []
        for idp in idps:
            st = _style_fetch(stylebank, idp, {'marker':'o','color':'#1f77b4'})
            h = plt.Line2D([], [], marker=st.get('marker','o'),
                        color=st.get('color','#1f77b4'),
                        linestyle='None', markersize=8,
                        markeredgecolor=_tone_hex(st.get('color','#1f77b4'),0.6))
            handles.append(h)
        ax1.legend(handles, idps, bbox_to_anchor=(1.02,0.5), loc='center left', fontsize=8, title='IDP', frameon=True)
        fig1.subplots_adjust(right=0.72)

    fig1.subplots_adjust(bottom=0.26)
    fig1.text(0.5, 0.02,
              "Interpretation: Higher values indicate better preservation of subject order",
              ha='center', fontsize=max(8,mpl.rcParams.get("font.size",10)-INTERP_FS_DELTA),
              color='blue', style='italic')

    figs.append(("pairwise_spearman_idpavg_combined", fig1))
     
    # -------------------------
    # Timepair average boxplot + scatter
    # -------------------------
    boxdata_tp = [[per_dir_tp_avgs[i].get(tp, np.nan) for tp in tps] for i in range(len(per_dir_tp_avgs))]
    fig2, ax2 = plt.subplots(figsize=(max(8,1.2*len(labels)),6)); tidy_axes(ax2)
    ax2.boxplot([ [v for v in arr if not np.isnan(v)] for arr in boxdata_tp],
                positions=np.arange(1,len(labels)+1), widths=0.6, patch_artist=True,
                boxprops=dict(facecolor='none'))

    # scatter per timepair
    for i, d in enumerate(per_dir_tp_avgs):
        for tp in tps:
            val = d.get(tp, np.nan)
            if np.isnan(val): 
                continue
            st = _style_fetch(stylebank, tp, {'marker':'o','color':'#1f77b4'})
            ax2.scatter(i+1 + np.random.uniform(-0.08,0.08), val,
                        marker=st.get('marker','o'), color=st.get('color','#1f77b4'),
                        edgecolor=_tone_hex(st.get('color','#1f77b4'),0.6), s=60)

    # Timepair legend
    if len(tps) > 0 and len(tps) <= 40:
        handles = []
        for tp in tps:
            st = _style_fetch(stylebank, tp, {'marker':'o','color':'#1f77b4'})
            h = plt.Line2D([], [], marker=st.get('marker','o'),
                        color=st.get('color','#1f77b4'),
                        linestyle='None', markersize=8,
                        markeredgecolor=_tone_hex(st.get('color','#1f77b4'),0.6))
            handles.append(h)
        ax2.legend(handles, tps, bbox_to_anchor=(1.02,0.5), loc='center left', fontsize=8, title='TimePair', frameon=True)
        fig2.subplots_adjust(right=0.72)


    fig2.subplots_adjust(bottom=0.26)
    fig2.text(0.5, 0.02,
              "Interpretation: Higher values indicate better preservation of subject order",
              ha='center', fontsize=max(8,mpl.rcParams.get("font.size",10)-INTERP_FS_DELTA),
              color='blue', style='italic')

    figs.append(("pairwise_spearman_timepairavg_combined", fig2))

    return figs
