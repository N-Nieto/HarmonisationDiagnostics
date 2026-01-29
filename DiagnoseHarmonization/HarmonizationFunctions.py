"""

Script containing self contained harmonization functions that can be used in conjunction with the diagnostic tools:

ComBat:
    Run ComBat harmonization on the data and return the harmonized data:
lme_harmonization:
    Run harmonization via linear mixed effects model on the data and return the residualized data
CovBat:
    Run CovBat harmonization on the data and return the harmonized data

"""

import numpy as np
import pandas as pd
import pandas as pd
import patsy
import sys
import numpy.linalg as la
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def combat_temp_for_covbat(data, batch, model=None, numerical_covariates=None, eb=True):
    """Correct for batch effects in a dataset

    Parameters
    ----------
    data : pandas.DataFrame
        A (n_features, n_samples) dataframe of the expression or methylation
        data to batch correct
    batch : pandas.Series
        A column corresponding to the batches in the data, with index same as
        the columns that appear in ``data``
    model : patsy.design_info.DesignMatrix, optional
        A model matrix describing metadata on the samples which could be
        causing batch effects. If not provided, then will attempt to coarsely
        correct just from the information provided in ``batch``
    numerical_covariates : list-like
        List of covariates in the model which are numerical, rather than
        categorical
    eb : logical
        Should empirical Bayes adjustments be made, if FALSE then gamma_hat
        and delta_hat are used as correction

    Returns
    -------
    corrected : pandas.DataFrame
        A (n_features, n_samples) dataframe of the batch-corrected data
    """
    if isinstance(numerical_covariates, str):
        numerical_covariates = [numerical_covariates]
    if numerical_covariates is None:
        numerical_covariates = []

    if model is not None and isinstance(model, pd.DataFrame):
        model["batch"] = list(batch)
    else:
        model = pd.DataFrame({'batch': batch})

    batch_items = model.groupby("batch").groups.items()
    batch_levels = [k for k, v in batch_items]
    batch_info = [v for k, v in batch_items]
    n_batch = len(batch_info)
    n_batches = np.array([len(v) for v in batch_info])
    n_array = float(sum(n_batches))

    # drop intercept
    drop_cols = [cname for cname, inter in  ((model == 1).all()).items() if inter == True]
    drop_idxs = [list(model.columns).index(cdrop) for cdrop in drop_cols]
    model = model[[c for c in model.columns if not c in drop_cols]]
    numerical_covariates = [list(model.columns).index(c) if isinstance(c, str) else c
            for c in numerical_covariates if not c in drop_cols]

    design = design_mat(model, numerical_covariates, batch_levels)

    sys.stderr.write("Standardizing Data across genes.\n")
    B_hat = np.dot(np.dot(la.inv(np.dot(design.T, design)), design.T), data.T)
    grand_mean = np.dot((n_batches / n_array).T, B_hat[:n_batch,:])
    var_pooled = np.dot(((data - np.dot(design, B_hat).T)**2), np.ones((int(n_array), 1)) / int(n_array))

    stand_mean = np.dot(grand_mean.T.reshape((len(grand_mean), 1)), np.ones((1, int(n_array))))
    tmp = np.array(design.copy())
    tmp[:,:n_batch] = 0
    stand_mean  += np.dot(tmp, B_hat).T

    s_data = ((data - stand_mean) / np.dot(np.sqrt(var_pooled), np.ones((1, int(n_array)))))

    sys.stderr.write("Fitting L/S model and finding priors\n")
    batch_design = design[design.columns[:n_batch]]
    gamma_hat = np.dot(np.dot(la.inv(np.dot(batch_design.T, batch_design)), batch_design.T), s_data.T)

    delta_hat = []

    for i, batch_idxs in enumerate(batch_info):
        #batches = [list(model.columns).index(b) for b in batches]
        delta_hat.append(s_data[batch_idxs].var(axis=1))

    gamma_bar = gamma_hat.mean(axis=1) 
    t2 = gamma_hat.var(axis=1)
   

    a_prior = list(map(aprior, delta_hat))
    b_prior = list(map(bprior, delta_hat))

    sys.stderr.write("Finding parametric adjustments\n")
    gamma_star, delta_star = [], []
    for i, batch_idxs in enumerate(batch_info):
        #print '18 20 22 28 29 31 32 33 35 40 46'
        #print batch_info[batch_id]

        temp = it_sol(s_data[batch_idxs], gamma_hat[i],
                     delta_hat[i], gamma_bar[i], t2[i], a_prior[i], b_prior[i])

        gamma_star.append(temp[0])
        delta_star.append(temp[1])

    sys.stdout.write("Adjusting data\n")
    bayesdata = s_data
    gamma_star = np.array(gamma_star)
    delta_star = np.array(delta_star)


    for j, batch_idxs in enumerate(batch_info):
        if eb:
            dsq = np.sqrt(delta_star[j,:])
            dsq = dsq.reshape((len(dsq), 1))
            denom =  np.dot(dsq, np.ones((1, n_batches[j])))
            numer = np.array(bayesdata[batch_idxs] - np.dot(batch_design.loc[batch_idxs], gamma_star).T)

            bayesdata[batch_idxs] = numer / denom
        else:
            gamma_hat = np.array(gamma_hat)
            delta_hat = np.array(delta_hat)
            
            dsq = np.sqrt(delta_hat[j,:])
            dsq = dsq.reshape((len(dsq), 1))
            denom =  np.dot(dsq, np.ones((1, n_batches[j])))
            numer = np.array(bayesdata[batch_idxs] - np.dot(batch_design.loc[batch_idxs], gamma_hat).T)

            bayesdata[batch_idxs] = numer / denom

    vpsq = np.sqrt(var_pooled).reshape((len(var_pooled), 1))
    bayesdata = bayesdata * np.dot(vpsq, np.ones((1, int(n_array)))) + stand_mean
 
    return bayesdata


def design_mat(mod, numerical_covariates, batch_levels):
    # require levels to make sure they are in the same order as we use in the
    # rest of the script.
    design = patsy.dmatrix("~ 0 + C(batch, levels=%s)" % str(batch_levels),
                                                  mod, return_type="dataframe")

    mod = mod.drop(["batch"], axis=1)
    numerical_covariates = list(numerical_covariates)
    sys.stderr.write("found %i batches\n" % design.shape[1])
    other_cols = [c for i, c in enumerate(mod.columns)
                  if not i in numerical_covariates]
    factor_matrix = mod[other_cols]
    design = pd.concat((design, factor_matrix), axis=1)
    if numerical_covariates is not None:
        sys.stderr.write("found %i numerical covariates...\n"
                            % len(numerical_covariates))
        for i, nC in enumerate(numerical_covariates):
            cname = mod.columns[nC]
            sys.stderr.write("\t{0}\n".format(cname))
            design[cname] = mod[mod.columns[nC]]
    sys.stderr.write("found %i categorical variables:" % len(other_cols))
    sys.stderr.write("\t" + ", ".join(other_cols) + '\n')
    return design

# --------------------- Placeholder helper functions ---------------------
# Translated from MATLAB, need to have concistency checked with NeuroComBat
def aprior(delta_hat):
    m = np.mean(delta_hat)
    s2 = np.var(delta_hat,ddof=1)
    return (2 * s2 +m**2) / float(s2)

def bprior(delta_hat):
    m = delta_hat.mean()
    s2 = np.var(delta_hat,ddof=1)
    return (m*s2+m**3)/s2

def postmean(g_hat, g_bar, n, d_star, t2):
    return (t2*n*g_hat+d_star * g_bar) / (t2*n+d_star)

def postvar(sum2, n, a, b):
    return (0.5 * sum2 + b) / (n / 2.0 + a - 1.0)

def itSol(sdat_batch, gamma_hat, delta_hat, gamma_bar, t2, a, b, conv=0.001):

    g_old = gamma_hat
    d_old = delta_hat
    change = 1
    count = 0
    while change > conv:
        g_new = postmean(gamma_hat, gamma_bar, sdat_batch.shape[1], d_old, t2)
        sum2 = np.sum((sdat_batch - g_new[:, None]) ** 2, axis=1)
        d_new = postvar(sum2, sdat_batch.shape[1], a, b)
        change = max(np.max(np.abs(g_new - g_old) / (np.abs(g_old) + 1e-8)),
                     np.max(np.abs(d_new - d_old) / (np.abs(d_old) + 1e-8)))
        g_old = g_new
        d_old = d_new
        count += 1
        if count > 100:
            print('Warning: itSol did not converge after 100 iterations')
            break
    return np.vstack([g_new, d_new])

def it_sol(sdat, g_hat, d_hat, g_bar, t2, a, b, conv=0.0001):
    n = (1 - np.isnan(sdat)).sum(axis=1)
    g_old = g_hat.copy()
    d_old = d_hat.copy()

    change = 1
    count = 0
    while change > conv:
        #print g_hat.shape, g_bar.shape, t2.shape
        g_new = postmean(g_hat, g_bar, n, d_old, t2)
        sum2 = ((sdat - np.dot(g_new.values.reshape((g_new.shape[0], 1)), np.ones((1, sdat.shape[1])))) ** 2).sum(axis=1)
        d_new = postvar(sum2, n, a, b)
       
        change = max((abs(g_new - g_old) / g_old).max(), (abs(d_new - d_old) / d_old).max())
        g_old = g_new #.copy()
        d_old = d_new #.copy()
        count = count + 1
    adjust = (g_new, d_new)
    return adjust 

def adjust_nums(numerical_covariates, drop_idxs):
    # if we dropped some values, have to adjust those with a larger index.
    if numerical_covariates is None: return drop_idxs
    return [nc - sum(nc < di for di in drop_idxs) for nc in numerical_covariates]

# ----------------------------- Main functions -----------------------------
# Define ComBat harmonization function
def combat(data, batch, mod, parametric,
           DeltaCorrection=True, UseEB=True, ReferenceBatch=None,
           RegressCovariates=False, GammaCorrection=True):
    """
    Run ComBat harmonization on the data and return the harmonized data.

    This version accepts numpy arrays or pandas DataFrame/Series for data, batch, and mod.
    If a DataFrame is supplied, columns are treated as samples (so data.shape == (n_features, n_samples)).
    The function will auto-transpose data or mod if it detects that samples were provided as rows.
    The returned bayesdata is the same type as input data (DataFrame -> DataFrame, ndarray -> ndarray).

    Note: helper functions aprior, bprior, itSol must be defined in scope.
    """

    # Remember whether inputs were pandas objects so we can restore types/labels on output
    dat_was_df = isinstance(data, pd.DataFrame)
    batch_was_series = isinstance(batch, (pd.Series, pd.Index))
    mod_was_df = isinstance(mod, pd.DataFrame)

    # Keep original labels (if any) to restore later
    dat_orig_index = data.index if dat_was_df else None
    dat_orig_columns = data.columns if dat_was_df else None
    batch_index = batch.index if batch_was_series else None
    mod_orig_index = mod.index if mod_was_df else None
    mod_orig_columns = mod.columns if mod_was_df else None

    # Convert pandas -> numpy, but allow transposing if the user supplied samples as rows
    # For data: desired internal shape = (n_features, n_samples) (rows=features, cols=samples)
    if dat_was_df:
        dat_np = data.values.astype(float)
        # if batch length matches number of rows, assume user gave samples as rows and transpose
        len_batch = len(batch)
        if len_batch == dat_np.shape[0] and len_batch != dat_np.shape[1]:
            dat_np = dat_np.T
            dat_transposed = True
        else:
            dat_transposed = False
    else:
        dat_np = np.asarray(data, dtype=float)
        dat_transposed = False

    # Normalize batch into 1D numpy array
    if batch_was_series:
        batch_np = batch.values
    else:
        batch_np = np.asarray(batch)
    batch_np = batch_np.ravel()

    # If dat_np and batch lengths mismatch, try to detect transposed data (samples as rows)
    if dat_np.ndim != 2:
        raise ValueError('Data matrix "data" must be 2-dimensional (features x samples).')

    # If batch length matches rows instead of columns, transpose dat_np
    if batch_np.shape[0] == dat_np.shape[0] and batch_np.shape[0] != dat_np.shape[1]:
        dat_np = dat_np.T
        dat_transposed = not dat_transposed  # flip if we already flipped earlier

    # Now dat_np shape[1] should equal batch length
    if dat_np.shape[1] != batch_np.shape[0]:
        raise ValueError('Number of samples in "data" must match length of "batch" vector.')

    # Handle mod (design covariates). Desired internal shape: (n_samples, n_covariates)
    if mod is None:
        mod_np = None
    else:
        if mod_was_df:
            mod_np = mod.values.astype(float)
            # If mod rows equal n_samples -> OK; else if mod.columns equal n_samples -> transpose
            n_samples = dat_np.shape[1]
            if mod_np.shape[0] == n_samples:
                pass
            elif mod_np.shape[1] == n_samples:
                mod_np = mod_np.T
            else:
                raise ValueError('Design matrix "mod" shape not compatible with data samples.')
        else:
            mod_np = np.asarray(mod, dtype=float)
            if mod_np.ndim == 1:
                # single covariate vector
                if mod_np.shape[0] == dat_np.shape[1]:
                    mod_np = mod_np.reshape(-1, 1)
                elif mod_np.shape[0] == dat_np.shape[0]:
                    # maybe passed per-feature by mistake
                    mod_np = mod_np.reshape(1, -1).T
                else:
                    raise ValueError('Design matrix "mod" length is incompatible with number of samples.')
            else:
                # 2D array: check orientation
                if mod_np.shape[0] != dat_np.shape[1] and mod_np.shape[1] == dat_np.shape[1]:
                    # mod provided as (n_covariates x n_samples) -> transpose
                    mod_np = mod_np.T
                elif mod_np.shape[0] != dat_np.shape[1] and mod_np.shape[1] != dat_np.shape[1]:
                    raise ValueError('Design matrix "mod" rows must match number of samples in "data".')

    # Use these working arrays from now on
    data = dat_np
    batch = batch_np
    mod = mod_np

    # Check the given parameters and print status messages
    if ReferenceBatch is None:
        print('Reference batch not given, defaulting to no reference')
    else:
        print(f'ReferenceBatch = {ReferenceBatch} -- fitting prior estimates using this batch and leaving batch unchanged')

    if not UseEB:
        print('Empirical Bayes set to false, using first estimates from raw mean and variances')
    else:
        print('Empirical Bayes set to true')

    if RegressCovariates:
        print('Regress Covariates set to true, skipping re-addition of OLS covariate estimates ')

    if not DeltaCorrection:
        print('Delta correction set to False, applying no delta (scale) correction on data')

    if not GammaCorrection:
        print('Gamma correction set to False, applying no gamma (mean) correction on data')

    # Basic input validation (after conversions)
    if data.ndim != 2:
        raise ValueError('Data matrix "data" must be 2-dimensional (features x samples).')
    if batch.ndim != 1:
        raise ValueError('Batch vector "batch" must be 1-dimensional (samples,).')
    if data.shape[1] != batch.shape[0]:
        raise ValueError('Number of samples in "data" must match length of "batch" vector.')
    if mod is not None:
        if mod.ndim != 2:
            raise ValueError('Design matrix "mod" must be 2-dimensional (samples x covariates).')
        if mod.shape[0] != data.shape[1]:
            raise ValueError('Number of samples in "data" must match number of rows in "mod" design matrix.')

    # --------------------- Begin ComBat core logic ---------------------

    # Compute SDs across samples for each feature (row)
    sds = np.std(data, axis=1, ddof=1)
    wh = np.where(sds == 0)[0]
    if wh.size > 0:
        raise ValueError('Error. There are rows with constant values across samples. Remove these rows and rerun ComBat.')

    # Convert batch vector to categorical and create dummy variables
    batch_cat = pd.Categorical(batch)
    batchmod = pd.get_dummies(batch_cat, drop_first=False).values  # shape (n_samples, n_batch)

    # Number of batches
    n_batch = batchmod.shape[1]
    levels = np.array(batch_cat.categories)
    print(f'[combat] Found {n_batch} batches')

    # Create list of arrays each containing sample indices for a batch
    batches = [np.where(batch == lev)[0] for lev in levels]

    # Size of each batch and total number of samples
    n_batches = np.array([len(b) for b in batches])
    n_array = np.sum(n_batches)

    # Construct design matrix including batch and additional covariates (mod)
    if mod is None:
        mod_arr = np.zeros((data.shape[1], 0))
    else:
        mod_arr = np.asarray(mod, dtype=float)
        if mod_arr.ndim == 1:
            mod_arr = mod_arr.reshape(-1, 1)

    design = np.hstack([batchmod, mod_arr])  # shape (n_samples, n_batch + n_cov)

    # Remove intercept column if present
    intercept = np.ones((n_array, 1))
    cols_to_keep = []
    for j in range(design.shape[1]):
        if not np.allclose(design[:, j], intercept.ravel()):
            cols_to_keep.append(j)
    design = design[:, cols_to_keep]

    print(f'[combat] Adjusting for {design.shape[1] - n_batch} covariate(s) of covariate level(s)')

    # Check for confounding between batch and covariates
    if np.linalg.matrix_rank(design) < design.shape[1]:
        nn = design.shape[1]
        if nn == (n_batch + 1):
            raise ValueError('Error. The covariate is confounded with batch. Remove the covariate and rerun ComBat.')
        if nn > (n_batch + 1):
            temp = design[:, (n_batch):nn]
            if np.linalg.matrix_rank(temp) < temp.shape[1]:
                raise ValueError('Error. The covariates are confounded. Please remove one or more of the covariates so the design is not confounded.')
            else:
                raise ValueError('Error. At least one covariate is confounded with batch. Please remove confounded covariates and rerun ComBat.')

    print('[combat] Standardizing Data across features')

    # Estimate coefficients B_hat using least squares: B_hat = inv(design' * design) * design' * data'
    XtX = design.T @ design
    inv_XtX = np.linalg.pinv(XtX) # Find the pseudo-inverse in case XtX is singular
    B_hat = inv_XtX @ design.T @ data.T  # shape (k, n_features) 

    # Reference batch handling
    if ReferenceBatch is not None:
        try:
            ref_idx = int(np.where(levels == ReferenceBatch)[0][0])
        except Exception:
            raise ValueError('ReferenceBatch not found in batch levels.')

        ref_samples = batches[ref_idx]
        ref_batch_effect = B_hat[ref_idx, :]

        if design.shape[1] > n_batch:
            tmp = design.copy()
            tmp[:, :n_batch] = 0
            Cov_effects = (tmp @ B_hat).T
        else:
            Cov_effects = np.zeros((data.shape[0], data.shape[1]))

        design_ref = design[ref_samples, :]
        predicted_ref = (design_ref @ B_hat).T
        residuals_ref = data[:, ref_samples] - predicted_ref
        var_ref = np.mean(residuals_ref ** 2, axis=1)

        stand_mean = np.tile(ref_batch_effect[:, None], (1, n_array))
        stand_mean = stand_mean + Cov_effects
        var_pooled = var_ref.copy()
        print(f'The size of the var_pooled array is {var_pooled.shape}')
    else:
        n_features = data.shape[0]
        n_samples = data.shape[1]
        XtX = design.T @ design
        inv_XtX = np.linalg.pinv(XtX)
        B_hat = inv_XtX @ design.T @ data.T
        grand_mean = (n_batches / n_array) @ B_hat[0:n_batch, :]
        predicted = (design @ B_hat).T
        resid = data - predicted
        var_pooled = np.mean(resid ** 2, axis=1)
        if np.any(var_pooled == 0):
            nonzeros = var_pooled[var_pooled != 0]
            if nonzeros.size > 0:
                var_pooled[var_pooled == 0] = np.median(nonzeros)
            else:
                var_pooled[var_pooled == 0] = 1e-6

        stand_mean = np.tile(grand_mean[:, None], (1, n_array))
        if design.shape[1] > n_batch:
            tmp = design.copy()
            tmp[:, :n_batch] = 0
            stand_mean = stand_mean + (tmp @ B_hat).T

    # Optional: regress covariates
    if design.shape[1] > n_batch:
        X_cov = design[:, n_batch:]
        X_cov = X_cov - np.mean(X_cov, axis=0, keepdims=True)
        B_cov = B_hat[n_batch:, :]
        Cov_effects = (X_cov @ B_cov).T
    else:
        Cov_effects = np.zeros_like(data)

    # Standardize the data, adding in small constant to avoid division by zero
    s_data = (data - stand_mean) / (np.sqrt(var_pooled)[:, None] + 1e-8)

    # Estimate batch effect parameters using least squares
    print('[combat] Fitting L/S model and finding priors')
    batch_design = design[:, :n_batch]  # samples x n_batch
    XtX_b = batch_design.T @ batch_design
    inv_XtX_b = np.linalg.pinv(XtX_b)
    gamma_hat = inv_XtX_b @ batch_design.T @ s_data.T  # shape (n_batch, n_features)
    print(f'Size of gamma hat: {gamma_hat.shape}')

    # Estimate batch-specific variances
    delta_hat = np.zeros((n_batch, data.shape[0]))
    for i in range(n_batch):
        indices = batches[i]
        if len(indices) > 1:
            delta_hat[i, :] = np.var(s_data[:, indices], axis=1, ddof=1)
        else:
            delta_hat[i, :] = np.var(s_data[:, indices], axis=1, ddof=0) + 1e-6

    print(f'Size of delta hat: {delta_hat.shape}')

    # Compute hyperparameters
    gamma_bar = np.mean(gamma_hat, axis=1)
    t2 = np.var(gamma_hat, axis=1, ddof=1)
    t2[t2 == 0] = 1e-6

    a_prior = np.zeros(n_batch)
    b_prior = np.zeros(n_batch)
    for i in range(n_batch):
        a_prior[i] = aprior(delta_hat[i, :])
        b_prior[i] = bprior(delta_hat[i, :])

    # Apply empirical Bayes estimates (parametric)
    if parametric:
        print('[combat] Finding parametric adjustments')
        gamma_star = np.zeros_like(gamma_hat)
        delta_star = np.zeros_like(delta_hat)

        for i in range(n_batch):
            indices = batches[i]
            if len(indices) == 0:
                continue
            temp = itSol(s_data[:, indices], gamma_hat[i, :], delta_hat[i, :],
                         gamma_bar[i], t2[i], a_prior[i], b_prior[i], conv=0.001)
            gamma_star[i, :] = temp[0, :]
            delta_star[i, :] = temp[1, :]

        if ReferenceBatch is not None:
            gamma_star[ref_idx, :] = np.zeros(data.shape[0])
            delta_star[ref_idx, :] = np.ones(data.shape[0])
    else:
        gamma_star = gamma_hat.copy()
        delta_star = delta_hat.copy()

    print('Size of gamma_star:', gamma_star.shape)

    if not UseEB:
        print('Discounting the EB adjustments and using Raw estimates, this is not advised')
        delta_star = delta_hat.copy()
        gamma_star = gamma_hat.copy()

    # Apply the L/S adjustments to the standardized data
    print('[combat] Adjusting the Data')
    bayesdata = s_data.copy()

    if DeltaCorrection:
        if GammaCorrection:
            for i in range(n_batch):
                indices = batches[i]
                if len(indices) == 0:
                    continue
                bayesdata[:, indices] = (bayesdata[:, indices] - (gamma_star[i, :])[:, None]) / (np.sqrt(delta_star[i, :])[:, None] + 1e-8)
        else:
            for i in range(n_batch):
                indices = batches[i]
                if len(indices) == 0:
                    continue
                bayesdata[:, indices] = bayesdata[:, indices] / (np.sqrt(delta_star[i, :])[:, None] + 1e-8)
    else:
        if GammaCorrection:
            for i in range(n_batch):
                indices = batches[i]
                if len(indices) == 0:
                    continue
                bayesdata[:, indices] = (bayesdata[:, indices] - (gamma_star[i, :])[:, None])
        else:
            print('Warning: Both Gamma and delta have been set to false, no ComBat adjustments have been applied')

    # Transform data back to original scale
    if RegressCovariates:
        bayesdata = (bayesdata * (np.sqrt(var_pooled)[:, None])) + (stand_mean - Cov_effects)
    else:
        bayesdata = (bayesdata * (np.sqrt(var_pooled)[:, None])) + stand_mean

    # If input was a DataFrame, restore labels and original orientation
    if dat_was_df:
        # If we transposed the original data to get to (features x samples), we should map results back
        if dat_transposed:
            # original had samples as rows; so return shape (n_samples, n_features)
            bayes_out = pd.DataFrame(bayesdata.T, index=dat_orig_index, columns=dat_orig_columns)
        else:
            # original had features as rows and samples as columns -> keep (features x samples)
            bayes_out = pd.DataFrame(bayesdata, index=dat_orig_index, columns=dat_orig_columns)
        return bayes_out, delta_star, gamma_star

    # Otherwise return numpy arrays (as before)
    return bayesdata, delta_star, gamma_star
# Define CovBat harmonization function: from Chen et al. 2022
def covbat(data, batch, model=None, numerical_covariates=None, pct_var=0.95, n_pc=0):

    """Correction of *Cov*ariance *Bat* effects

    Parameters
    ----------
    data : pandas.DataFrame
        A (n_features, n_samples) dataframe of the expression or methylation
        data to batch correct
    batch : pandas.Series
        A column corresponding to the batches in the data, with index same as
        the columns that appear in ``data``
    model : patsy.design_info.DesignMatrix, optional
        A model matrix describing metadata on the samples which could be
        causing batch effects. If not provided, then will attempt to coarsely
        correct just from the information provided in ``batch``
    numerical_covariates : list-like
        List of covariates in the model which are numerical, rather than
        categorical
    pct_var : numeric
        Numeric between 0 and 1 indicating the percent of variation that is
        explained by the adjusted PCs
    n_pc : numeric
        If >0, then this specifies the number of PCs to adjust. Overrides pct_var

    Returns
    -------
    corrected : pandas.DataFrame
        A (n_features, n_samples) dataframe of the batch-corrected data
    """
    if isinstance(numerical_covariates, str):
        numerical_covariates = [numerical_covariates]
    if numerical_covariates is None:
        numerical_covariates = []

    if model is not None and isinstance(model, pd.DataFrame):
        model["batch"] = list(batch)
    else:
        model = pd.DataFrame({'batch': batch})

    batch_items = model.groupby("batch").groups.items()
    batch_levels = [k for k, v in batch_items]
    batch_info = [v for k, v in batch_items]
    n_batch = len(batch_info)
    n_batches = np.array([len(v) for v in batch_info])
    n_array = float(sum(n_batches))

    # drop intercept
    drop_cols = [cname for cname, inter in  ((model == 1).all()).items() if inter == True]
    drop_idxs = [list(model.columns).index(cdrop) for cdrop in drop_cols]
    model = model[[c for c in model.columns if not c in drop_cols]]
    numerical_covariates = [list(model.columns).index(c) if isinstance(c, str) else c
        for c in numerical_covariates if not c in drop_cols]

    design = design_mat(model, numerical_covariates, batch_levels)

    sys.stderr.write("Standardizing Data across genes.\n")
    B_hat = np.dot(np.dot(la.inv(np.dot(design.T, design)), design.T), data.T)
    grand_mean = np.dot((n_batches / n_array).T, B_hat[:n_batch,:])
    var_pooled = np.dot(((data - np.dot(design, B_hat).T)**2), np.ones((int(n_array), 1)) / int(n_array))

    stand_mean = np.dot(grand_mean.T.reshape((len(grand_mean), 1)), np.ones((1, int(n_array))))
    tmp = np.array(design.copy())
    tmp[:,:n_batch] = 0
    stand_mean  += np.dot(tmp, B_hat).T

    s_data = ((data - stand_mean) / np.dot(np.sqrt(var_pooled), np.ones((1, int(n_array)))))

    sys.stderr.write("Fitting L/S model and finding priors\n")
    batch_design = design[design.columns[:n_batch]]
    gamma_hat = np.dot(np.dot(la.inv(np.dot(batch_design.T, batch_design)), batch_design.T), s_data.T)

    delta_hat = []

    for i, batch_idxs in enumerate(batch_info):
        #batches = [list(model.columns).index(b) for b in batches]
        delta_hat.append(s_data[batch_idxs].var(axis=1))

    gamma_bar = gamma_hat.mean(axis=1) 
    t2 = gamma_hat.var(axis=1)
   
    a_prior = list(map(aprior, delta_hat))
    b_prior = list(map(bprior, delta_hat))

    sys.stderr.write("Finding parametric adjustments\n")
    gamma_star, delta_star = [], []
    for i, batch_idxs in enumerate(batch_info):
        #print '18 20 22 28 29 31 32 33 35 40 46'
        #print batch_info[batch_id]

        temp = it_sol(s_data[batch_idxs], gamma_hat[i],
                     delta_hat[i], gamma_bar[i], t2[i], a_prior[i], b_prior[i])

        gamma_star.append(temp[0])
        delta_star.append(temp[1])

    sys.stdout.write("Adjusting data\n")
    bayesdata = s_data
    gamma_star = np.array(gamma_star)
    delta_star = np.array(delta_star)

    for j, batch_idxs in enumerate(batch_info):

        dsq = np.sqrt(delta_star[j,:])
        dsq = dsq.reshape((len(dsq), 1))
        denom =  np.dot(dsq, np.ones((1, n_batches[j])))
        numer = np.array(bayesdata[batch_idxs] - np.dot(batch_design.loc[batch_idxs], gamma_star).T)

        bayesdata[batch_idxs] = numer / denom
   
    vpsq = np.sqrt(var_pooled).reshape((len(var_pooled), 1))
    # not adding back stand_mean yet
    bayesdata = bayesdata * np.dot(vpsq, np.ones((1, int(n_array))))

    # CovBat step: PCA then ComBat without EB on the scores
    # comdata = data.T
    comdata = bayesdata.T
    bmu = np.mean(comdata, axis=0)
    # standardize data before PCA
    scaler = StandardScaler()
    comdata = scaler.fit_transform(comdata)
    
    pca = PCA()
    pca.fit(comdata)
    pc_comp = pca.components_
    full_scores = pd.DataFrame(pca.fit_transform(comdata)).T
    full_scores.columns = data.columns

    var_exp=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4))
    npc = np.min(np.where(var_exp>pct_var))+1
    if n_pc > 0:
        npc = n_pc
    scores = full_scores.loc[range(0,npc),:]
    scores_com = combat_temp_for_covbat(scores, batch, model=None, eb=False)
    full_scores.loc[range(0,npc),:] = scores_com

    x_covbat = bayesdata - bayesdata # create pandas DataFrame to store output
    # x_covbat = x_covbat.add(bmu, axis='index')
    proj = np.dot(full_scores.T, pc_comp).T
    x_covbat += scaler.inverse_transform(proj.T).T
    # x_covbat = x_covbat * np.dot(vpsq, np.ones((1, int(n_array)))) + stand_mean
    x_covbat += stand_mean
 
    return x_covbat
# Define harmonization via mixed effects model (Regression analysis)
import numpy as np
import pandas as pd
import warnings
from statsmodels.formula.api import mixedlm

def lme_harmonization(data, batch, mod, variable_names):
    """
    Fit per-feature linear mixed-effects models and return residualised data.

    Accepts numpy arrays or pandas DataFrame/Series:
      - data: samples x features  (np.ndarray) or pandas.DataFrame (rows=samples, cols=features)
      - batch: length-n_samples array-like or pandas.Series with batch labels (grouping factor)
      - mod: n_samples x n_covariates (np.ndarray) or pandas.DataFrame
      - variable_names: list of length n_covariates naming covariates (strings)

    Returns:
      - residuals: same type as input `data` (DataFrame if DataFrame input, ndarray otherwise),
                   with shape (n_samples, n_features)

    Notes:
      - This fits a mixed-effects model separately for each feature/column in `data`.
      - Formula used for each feature: Y ~ batch + covariates + batch:cov1 + batch:cov2 + ...
        Random intercept for batch (groups=batch) is used.
      - The residuals returned have batch (and covariate) effects removed.
    """
    # ----------------------------
    # Normalize inputs & keep labels
    # ----------------------------
    data_was_df = isinstance(data, pd.DataFrame)
    batch_was_series = isinstance(batch, (pd.Series, pd.Index))
    mod_was_df = isinstance(mod, pd.DataFrame)

    # keep labels to restore later
    data_index = data.index if data_was_df else None
    data_columns = data.columns if data_was_df else None

    # Convert inputs to numpy arrays of expected orientation:
    # internal working shape for `data_np` is (n_samples, n_features)
    if data_was_df:
        data_np = data.values.astype(float)
    else:
        data_np = np.asarray(data, dtype=float)

    # batch -> 1D array of length n_samples
    if batch_was_series:
        batch_np = np.asarray(batch.values)
    else:
        batch_np = np.asarray(batch).ravel()

    # mod -> (n_samples, n_covariates) or None
    if mod is None:
        mod_np = None
    else:
        if mod_was_df:
            mod_np = mod.values.astype(float)
        else:
            mod_np = np.asarray(mod, dtype=float)
        # if 1D, make column vector
        if mod_np.ndim == 1:
            mod_np = mod_np.reshape(-1, 1)

    # Basic validation
    if data_np.ndim != 2:
        raise ValueError("Data must be a 2D array (samples x features).")
    n_samples, n_features = data_np.shape

    # Check batch is numeric, if not convert to categorical codes
    if not np.issubdtype(batch_np.dtype, np.number):
        batch_cat = pd.Categorical(batch_np)
        batch_np = batch_cat.codes  # integer codes for categories

    if batch_np.ndim != 1 or batch_np.shape[0] != n_samples:
        print(batch_np.shape, n_samples)
        raise ValueError("Batch must be a 1D array-like with length equal to number of samples (rows of data).")

    if mod_np is not None:
        if mod_np.ndim != 2:
            raise ValueError("mod must be a 2D array (samples x covariates).")
        if mod_np.shape[0] != n_samples:
            raise ValueError("mod must have the same number of rows (samples) as data.")
        if len(variable_names) != mod_np.shape[1]:
            raise ValueError("variable_names length must equal number of covariates (columns of mod).")

    # ----------------------------
    # Build a base DataFrame with batch and covariates used for every per-feature fit
    # ----------------------------
    # We create a DataFrame with one row per sample. For each feature we will assign
    # the response column temporarily and fit a mixed model formula on that DataFrame.
    base_df = pd.DataFrame(index=range(n_samples))
    base_df['batch'] = batch_np  # grouping factor (categorical)

    if mod_np is not None:
        for i, var in enumerate(variable_names):
            base_df[var] = mod_np[:, i]

    # Prepare interaction terms string for the formula: batch:cov1 + batch:cov2 + ...
    interaction_terms = []
    if mod_np is not None:
        for var in variable_names:
            interaction_terms.append(f'batch:{var}')
    interaction_str = ' + '.join(interaction_terms) if interaction_terms else ''

    # Fixed part: batch + covariates
    fixed_parts = ['batch'] + (variable_names if variable_names else [])
    fixed_str = ' + '.join(fixed_parts)

    # full RHS for formula (skip empty pieces)
    if interaction_str:
        rhs = f"{fixed_str} + {interaction_str}"
    else:
        rhs = fixed_str

    # ----------------------------
    # Fit per-feature mixed-effects model and collect residuals
    # ----------------------------
    # We will fit: Y ~ <rhs> with groups=batch (random intercept).
    # This is done separately for each feature (column) in data_np.
    residuals = np.zeros_like(data_np, dtype=float)  # shape (n_samples, n_features)
    warnings.filterwarnings("ignore")  # suppress fit warnings; you may remove this

    for feat_idx in range(n_features):
        # create a temporary response column name that doesn't conflict with others
        resp_col = '_y_response'
        base_df[resp_col] = data_np[:, feat_idx]

        # formula example: '_y_response ~ batch + age + sex + batch:age + batch:sex'
        formula = f'Q("{resp_col}") ~ {rhs}'

        # Fit mixed model with random intercept for batch
        try:
            model = mixedlm(formula, base_df, groups=base_df['batch'])
            # try default fit; if convergence issues occur, fallback handled below
            result = model.fit()
        except Exception:
            # fallback fit with different options if default fails (method and reml off)
            model = mixedlm(formula, base_df, groups=base_df['batch'])
            result = model.fit(method='lbfgs', reml=False, maxiter=2000, disp=False)

        # result.resid is length n_samples
        residuals[:, feat_idx] = result.resid.values

        # drop temporary response column (next iteration will overwrite)
        base_df.drop(columns=[resp_col], inplace=True)

    # ----------------------------
    # Restore output type & labels to match input
    # ----------------------------
    if data_was_df:
        # preserve original index and column names
        residuals_df = pd.DataFrame(residuals, index=data_index if data_index is not None else range(n_samples),
                                    columns=data_columns if data_columns is not None else range(n_features))
        return residuals_df
    else:
        return residuals
# Define LME_IQM harmonization functions
def lme_iqm_harmonization(data, IQMs, mod, variable_names):
    # Run LME harmonization on IQMs to get resisuals:
    """
    inputs:
    - data: subjects x features (np.ndarray)
    - IQMs: subjects x IQMs (np.ndarray)
    - mod:  subjects x covariates (np.ndarray)
    - variable_names: covariates (list)
    returns:
    - residuals: subjects x features (np.ndarray)
    - params: dict of LME model parameters
    - original: subjects x features (np.ndarray)

    description: 
    This function attempts to fit a mixed effects model on the data using IQMs as mixed effects and covariates as fuxed effects.
    The residuals of the model are returned as the harmonized data, with covariate effects retained and IQM effects removed:



    """
    print("Place holder for future implementation of LME-IQM harmonization")

    return None