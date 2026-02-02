""" Test script for ComBat

Simulate a small dummy dataet and run the ComBat harmonization on it, printing the result in terminal.

Dataset structure is constructed to match the expectaions of ComBat

Y = alpha + X*beta_age +X*beta_sex + gamma_batch + delta_batch*epsilon

Here we simulate the dataset with two batches, one continuous covariate (age) and one binary (sex)

The data consists of 10 features (rows) and 20 samples (columns)

We create an array of random data of size feature x samples (10x20) drawn from a standard normal distribution centered at zero for each feature
, a feature mean of size (10x1), two covariates of size (20x1) respectively,

We then simulate a multiplicative batch effect (delta) and an additive batch effect (gamma)

The random array is multipled by the multiplicative batch effect and we then add the feature mean, covariate effects 
and additive batch effect to create the final dataset.


Here we can consider the final dataset to be the observed data convered to Z-scores for which we apply ComBat.
If ComBat runs correctly, we should see no errors.

"""

# Test combat

def test_ComBat_asarray():
    # Import necessary libraries
    import numpy as np
    import pandas as pd
    from DiagnoseHarmonization import HarmonizationFunctions

    # Create array of random data as numpy array
    np.random.seed(42)
    n_features = 10
    n_samples = 100
    data = np.random.randn(n_features, n_samples)
    # Add random batch effect:
    batch = np.array(['batch1'] * 50 + ['batch2'] * 50)
    # Add batch effect (simple mean shift for each batch)
    data[:, :50] += 1*np.random.randint(1, 3, size=(n_features, 50))  # Batch 1 has a mean shift of +2
    data[:, 50:] -= 1*np.random.randint(1, 3, size=(n_features, 50))  # Batch 2 has a mean shift of -2
    # Create an array with covariates (age, sex)
    age = np.random.randint(20, 60, size=n_samples)
    sex = np.random.randint(0, 2, size=n_samples)
    covariates = np.vstack((age, sex))
    # Run ComBat
    bayesdata, a,b = HarmonizationFunctions.combat(
        data,
        batch,
        covariates.T,
        parametric=True,
        DeltaCorrection=True,
        UseEB=True,
        ReferenceBatch=None
    )
    # Check that the output has the same shape as the input
    
    assert bayesdata.shape == data.shape

# Repeat logic above but with pandas DataFrame input
def test_ComBat_dataframe():
    # Import necessary libraries
    import numpy as np
    import pandas as pd
    from DiagnoseHarmonization import HarmonizationFunctions

    # Create array of random data as pandas DataFrame
    np.random.seed(42)
    n_features = 10
    n_samples = 100
    data = pd.DataFrame(
        np.random.randn(n_features, n_samples),
        index=[f"feature_{i}" for i in range(n_features)],
        columns=[f"sample_{i}" for i in range(n_samples)],
    )
    # Add random batch effect:
    batch = pd.Series(['batch1'] * 50 + ['batch2'] * 50, index=data.columns)
    # Add batch effect (simple mean shift for each batch)
    data.iloc[:, :50] += 1*np.random.randint(1, 3, size=(n_features, 50))  # Batch 1 has a mean shift of +2
    data.iloc[:, 50:] -= 1*np.random.randint(1, 3, size=(n_features, 50))  # Batch 2 has a mean shift of -2
    # Create a DataFrame with covariates (age, sex)
    age = np.random.randint(20, 60, size=n_samples) 
    sex = np.random.randint(0, 2, size=n_samples)
    covariates = pd.DataFrame(
        {
            "age": age,
            "sex": sex,
        },
        index=data.columns,
    )
    # Run ComBat
    bayesdata, a,b = HarmonizationFunctions.combat(
        data,
        batch,
        covariates,
        parametric=True,
        DeltaCorrection=True,
        UseEB=True,
        ReferenceBatch=None
    )
    # Check that the output has the same shape as the input
    assert bayesdata.shape == data.shape

def test_combat_as_covbat():
    # Import necessary libraries
    import numpy as np
    import pandas as pd
    from DiagnoseHarmonization import HarmonizationFunctions
    import numpy as np
    import pandas as pd
    from DiagnoseHarmonization import HarmonizationFunctions
    from sklearn.decomposition import PCA

    # Create array of random data as numpy array
    np.random.seed(42)
    n_features = 10
    n_samples = 100
    data = np.random.randn(n_features, n_samples)
    # Add random batch effect:
    batch = np.array(['batch1'] * 50 + ['batch2'] * 50)
    # Add batch effect (simple mean shift for each batch)
    data[:, :50] += 1*np.random.randint(1, 3, size=(n_features, 50))  # Batch 1 has a mean shift of +2
    data[:, 50:] -= 1*np.random.randint(1, 3, size=(n_features, 50))  # Batch 2 has a mean shift of -2
    # Create an array with covariates (age, sex)
    age = np.random.randint(20, 60, size=n_samples)
    sex = np.random.randint(0, 2, size=n_samples)
    covariates = np.vstack((age, sex))
    # Run ComBat with CovBat mode
    bayesdata, a,b = HarmonizationFunctions.combat(
        data,
        batch,
        covariates.T,
        parametric=True,
        DeltaCorrection=True,
        UseEB=True,
        ReferenceBatch=None,
        covbat_mode=True
    )
    # Check that the output has the same shape as the input
    assert bayesdata.shape == data.shape

def test_efficacy_com_cov():
    import numpy as np
    import pandas as pd
    from DiagnoseHarmonization import HarmonizationFunctions
    import numpy as np
    import pandas as pd
    from DiagnoseHarmonization import HarmonizationFunctions
    from sklearn.decomposition import PCA
    # Simulate data with batch effects
    np.random.seed(42)
    n_features = 50
    n_samples_per_batch = 30
    n_batches = 3
    total_samples = n_samples_per_batch * n_batches

    # Create batch-specific means and covariances
    batch_means = [np.random.randn(n_features) * 5 for _ in range(n_batches)]
    batch_covs = [np.diag(np.random.rand(n_features) + 0.5) for _ in range(n_batches)]

    data_list = []
    batch_labels = []
    for i in range(n_batches):
        batch_data = np.random.multivariate_normal(
            mean=batch_means[i],
            cov=batch_covs[i],
            size=n_samples_per_batch
        )
        data_list.append(batch_data)
        batch_labels.extend([f"batch_{i}"] * n_samples_per_batch)

    data = pd.DataFrame(
        np.vstack(data_list),
        index=[f"sample_{i}" for i in range(total_samples)],
        columns=[f"feature_{i}" for i in range(n_features)],
    ).T  # Transpose to (n_features, n_samples) 

    batch = pd.Series(batch_labels, index=data.columns)
    model = None
    numerical_covariates = []
    # Apply CovBat
    bayesdata, a,b = HarmonizationFunctions.combat(
    data,
    batch,
    mod=None,
    parametric=True,
    DeltaCorrection=True,
    UseEB=True,
    ReferenceBatch=None,
    covbat_mode=True
)

    pca = PCA(n_components=2)
    original_pca = pca.fit_transform(data.T)
    corrected_pca = pca.fit_transform(bayesdata.T)

    from sklearn.metrics import silhouette_score
    original_silhouette = silhouette_score(original_pca, batch)
    corrected_silhouette = silhouette_score(corrected_pca, batch)
    assert corrected_silhouette < original_silhouette, "CovBat did not reduce batch effects"
    
    # Produce plots to visualize PCA before and after correction
    # Create colours for batch for c= parameter in scatter
    batch_colors = batch.astype('category').cat.codes

    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(original_pca[:, 0], original_pca[:, 1], c=batch_colors)
    plt.title("PCA Before CovBat")
    plt.subplot(1, 2, 2)
    plt.scatter(corrected_pca[:, 0], corrected_pca[:, 1], c=batch_colors)
    plt.title("PCA After CovBat")
    plt.show()

    

if __name__ == "__main__":
    test_efficacy_com_cov()  