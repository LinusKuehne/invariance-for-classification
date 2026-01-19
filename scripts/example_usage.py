# setup logging to see library output
import logging

import numpy as np

from invariance_for_classification import StabilizedClassificationClassifier
from invariance_for_classification.generate_data.synthetic_DGP import generate_scm_data

logging.basicConfig(level=logging.INFO, format="%(name)s - %(message)s")

df, int_df = generate_scm_data(n_per_env=500, return_int_values=True)

# 2. Initialize the Stabilized Classification Classifier
clf = StabilizedClassificationClassifier(
    alpha_inv=0.05, alpha_pred=0.05, n_bootstrap=100, verbose=1
)

# 3. Fit (DataFrame input)

# Using dataframe input directly
# E is the environment index (not a feature).
clf.fit(df, y="Y", environment="E")

# 4. Inspect results

# Selected subsets:
if hasattr(clf, "active_subsets_"):
    for stat in clf.active_subsets_:
        subset_names = [f"Col{i}" for i in stat["subset"]]
        print(
            f"Subset Indices: {stat['subset']}, p-value: {stat['p_value']:.4f}, Score: {stat['score']:.4f}"
        )

# 5. Predict
# Test prediction on new data:
df_test, int_df_test = generate_scm_data(
    n_per_env=100, int_vals=[3.0], return_int_values=True
)
X_test = df_test.drop(columns=["E", "Y"]).values
y_test = df_test["Y"].values

preds = clf.predict(X_test)
acc = np.mean(preds == y_test)
print(f"Accuracy on unseen environment: {acc:.2f}")
