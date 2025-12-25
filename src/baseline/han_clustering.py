"""
Han et al. (2021) Baseline Clustering Pipeline

Steps:
1. Prepare Data: Pivot features to (Time, Ticker, Features)
2. PCA: Extract K principal components (Risk Factors)
3. Clustering: K-Means on the principal components
4. Evaluation: NMI against Ground Truth
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, OPTICS, AgglomerativeClustering
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from typing import Dict, List, Optional, Tuple

class HanClusteringPipeline:
    def __init__(self, method: str = 'kmeans', n_components: int = 5, n_clusters: int = 10,
                 outlier_percentile: float = 95.0, dist_quantile: float = None,
                 random_state: int = 42, **kwargs):
        self.method = method.lower()
        self.n_components = n_components
        self.n_clusters = n_clusters
        self.outlier_percentile = outlier_percentile
        self.dist_quantile = dist_quantile # New: Dynamic Threshold Quantile (KNN-based)
        self.random_state = random_state
        self.kwargs = kwargs

        # PCA handles the dimensionality reduction
        self.pca = PCA(n_components=n_components, random_state=random_state)

        # We need a scaler because PCA is sensitive to scale
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()

        self.is_fitted = False
        self.labels_ = None
        self.model = None

    def _init_model(self, X_sample: np.ndarray = None):
        """
        Initialize the clustering model.
        If X_sample is provided and self.dist_quantile is set, calculate dynamic threshold.
        Method: K-Dist Graph (KNN distances) - Canonical for DBSCAN/Clustering density.
        Metric: Manhattan (L1) as per Han et al. (2021).
        MinPts: ln(N) if not specified.
        """
        # Determine MinPts (min_samples)
        # Han et al: "MinPts is set to be the natural logarithm of the total number of data points N"
        min_samples_arg = self.kwargs.get('min_samples', None)

        if min_samples_arg is not None:
             min_samples = int(min_samples_arg)
        elif X_sample is not None and len(X_sample) > 0:
             # Log(N) rule
             N = len(X_sample)
             min_samples = int(np.log(N))
             # Ensure reasonable bounds (e.g., at least 3 to form a density)
             if min_samples < 3: min_samples = 3
             print(f"Dynamic MinPts (ln(N), N={N}): {min_samples}")
        else:
             min_samples = 5 # Fallback default

        # Dynamic Parameter Calculation (Threshold/Eps)
        dynamic_threshold = None

        if self.dist_quantile is not None and X_sample is not None and self.method in ['dbscan', 'agglomerative', 'optics']:
            from sklearn.neighbors import NearestNeighbors

            # Use k-th nearest neighbor (k = min_samples)
            # We need k+1 because the 1st NN is the point itself (dist=0)
            k = min_samples
            if k >= len(X_sample):
                k = len(X_sample) - 1

            # Metric: Manhattan (L1)
            nbrs = NearestNeighbors(n_neighbors=k+1, metric='manhattan').fit(X_sample)
            distances, _ = nbrs.kneighbors(X_sample)

            # The distance to the k-th nearest neighbor is in the last column
            k_distances = distances[:, -1] # Sorted ascending by distance in kneighbors result

            # Sort to find the quantile
            if len(k_distances) > 0:
                dynamic_threshold = np.percentile(k_distances, self.dist_quantile * 100)
                print(f"Dynamic Threshold (KNN-L1-dist q={self.dist_quantile}, k={k}): {dynamic_threshold:.4f}")

        # Kwargs Filtering
        kwargs_filtered = self.kwargs.copy()
        for k in ['eps', 'min_samples', 'distance_threshold']:
             kwargs_filtered.pop(k, None)

        if self.method == 'kmeans':
            # KMeans in sklearn is strictly Euclidean (L2).
            # To strictly support L1 we would need K-Medoids, but usually 'KMeans' implies L2.
            # We keep L2 for KMeans.
            self.model = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10, **kwargs_filtered)

        elif self.method == 'dbscan':
            eps = dynamic_threshold if dynamic_threshold else self.kwargs.get('eps', 0.5)
            # Explicitly set metric to manhattan
            self.model = DBSCAN(eps=eps, min_samples=min_samples, metric='manhattan', **kwargs_filtered)

        elif self.method == 'optics':
             # OPTICS uses max_eps? usually just min_samples
             if dynamic_threshold:
                 kwargs_filtered['max_eps'] = dynamic_threshold
             self.model = OPTICS(min_samples=min_samples, metric='manhattan', **kwargs_filtered)

        elif self.method == 'agglomerative':
             # Dynamic Logic: If dynamic_threshold set, use it and set n_clusters=None
             dist_thresh = dynamic_threshold if dynamic_threshold else self.kwargs.get('distance_threshold', None)
             n_clus = self.n_clusters

             if dist_thresh is not None:
                 n_clus = None
                 # AgglomerativeClustering requires n_clusters=None if distance_threshold given

             # Metric: Manhattan
             # Linkage: Ward does NOT support Manhattan. Must use 'average', 'complete', or 'single'.
             # Default to 'average' for robustness.
             linkage = self.kwargs.get('linkage', 'average')
             if linkage == 'ward': linkage = 'average' # Override if Ward was passed

             self.model = AgglomerativeClustering(n_clusters=n_clus, distance_threshold=dist_thresh, metric='manhattan', linkage=linkage, **kwargs_filtered)

        else:
            raise ValueError(f"Unknown clustering method: {self.method}")
        self.is_fitted = False

    def prepare_data(self, df: pd.DataFrame, is_training: bool = False) -> Tuple[np.ndarray, Optional[pd.Series], pd.Index]:
        """
        Convert feature DataFrame to matrix for clustering.
        Handles scaling: if training, fit_transform; else transform.
        """
        # Ensure numeric
        features = df.select_dtypes(include=[np.number])

        # Check for inf/nan
        features = features.replace([np.inf, -np.inf], np.nan).dropna()

        if features.empty:
            return np.array([]), None, pd.Index([])

        # Ground Truth intersection
        labels = None
        if 'industry' in df.columns:
            # Need to align labels with dropped NaNs
            labels = df.loc[features.index, 'industry']

        # Scaling
        if is_training:
            X_scaled = self.scaler.fit_transform(features)
        else:
            # If not fitted, this will raise NotFittedError, ensuring safe usage
            X_scaled = self.scaler.transform(features)

        return X_scaled, labels, features.index

    def fit(self, df: pd.DataFrame):
        """
        Fit PCA and Clustering on training data.
        """
        X_scaled, _, _ = self.prepare_data(df, is_training=True)
        if len(X_scaled) == 0:
            raise ValueError("No valid training data provided.")

        # 1. Fit PCA
        pca_features = self.pca.fit_transform(X_scaled)

        # 2. Init & Fit Model (Dynamic Support)
        # Pass pca_features to _init_model for dynamic threshold calculation
        self._init_model(X_sample=pca_features)

        try:
             self.model.fit(pca_features)
        except Exception as e:
             # Some models might fail on tiny data
             print(f"Clustering fit failed: {e}")
             return self

        self.is_fitted = True

        # 3. Get Initial Labels
        if hasattr(self.model, 'labels_'):
            labels = self.model.labels_.copy()
        else:
            # For models like AgglomerativeClustering without labels_ usually it exists but let's be safe
            labels = self.model.predict(pca_features) if hasattr(self.model, 'predict') else self.model.labels_

        # 4. Apply Density Filtering (Generic)
        self.labels_ = self._apply_density_filter(pca_features, labels)

        return self

    def _apply_density_filter(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Generic density filtering (Han et al. 2021).
        Calculates empirical centroids for any clustering method using MANHATTAN (L1) distance.
        """
        # If disabled
        if self.outlier_percentile >= 100 or self.outlier_percentile <= 0:
            self.outlier_mask_ = np.zeros(len(labels), dtype=bool)
            return labels

        min_dists = np.zeros(len(X))
        unique_labels = np.unique(labels)

        # Calculate distances to assigned empirical centroid
        for label in unique_labels:
            if label == -1:
                min_dists[labels == -1] = np.inf # Existing outliers
                continue

            mask = (labels == label)
            points = X[mask]

            if len(points) == 0: continue

            # Empirical Centroid
            # Note: For L1 distance, the geometric median is technically the minimizer,
            # but mean is often used as a robust proxy or if implied by "block distance to center".
            # We stick to Mean for centroid definition unless Median is strictly required.
            centroid = points.mean(axis=0)

            # Manhattan Distance (L1 Norm)
            # sum(|x - mu|)
            dvs = points - centroid
            d_i = np.sum(np.abs(dvs), axis=1)
            min_dists[mask] = d_i

        # Thresholding
        valid_dists = min_dists[min_dists != np.inf]

        if len(valid_dists) > 0:
            threshold = np.percentile(valid_dists, self.outlier_percentile)
            outliers = min_dists > threshold

            if outliers.sum() > 0:
                labels[outliers] = -1
                print(f"Density Filtering: Dropped {outliers.sum()} outliers (>{self.outlier_percentile}th %ile, L1-Dist).")
            self.outlier_mask_ = outliers
        else:
            self.outlier_mask_ = np.zeros(len(labels), dtype=bool)

        return labels

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """
        Predict clusters for new data.
        """
        if not self.is_fitted:
            raise RuntimeError("Pipeline not fitted. Call fit() first.")

        # If method is not K-Means, we cannot predict easily
        if self.method != 'kmeans':
             return pd.Series(dtype=float)

        X_scaled, _, valid_index = self.prepare_data(df, is_training=False)
        if len(X_scaled) == 0:
            return pd.Series(dtype=int)

        # 1. Transform PCA
        pca_features = self.pca.transform(X_scaled)

        # 2. Predict
        cluster_ids = self.model.predict(pca_features)

        return pd.Series(cluster_ids, index=valid_index, name='cluster')

    def train_clustering(self, df: pd.DataFrame) -> Dict:
        """
        Train (Fit) and Return Results.
        """
        self.fit(df)

        # Metrics using self.labels_
        features = df.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).dropna()

        if self.labels_ is None:
             return {'error': 'Fit failed'}

        cluster_labels = self.labels_

        # Metrics
        explained_variance = np.sum(self.pca.explained_variance_ratio_)
        metrics = {
            'n_samples': len(features),
            'n_features': features.shape[1],
            'explained_variance': explained_variance,
            # Inertia only for KMeans
            'inertia': getattr(self.model, 'inertia_', 0.0)
        }

        if 'industry' in df.columns:
            true_labels = df.loc[features.index, 'industry']
            metrics['nmi'] = normalized_mutual_info_score(true_labels, cluster_labels)
            metrics['ari'] = adjusted_rand_score(true_labels, cluster_labels)

        return {
            'metrics': metrics,
            'cluster_labels': pd.Series(cluster_labels, index=features.index, name='cluster'),
            'pca_components': self.pca.components_,
            'feature_names': features.columns.tolist()
        }

    def get_adjacency_matrix(self, cluster_series: pd.Series) -> pd.DataFrame:
        """
        Convert cluster labels to Block-Diagonal Adjacency Matrix on a SINGLE DAY.
        """
        tickers = cluster_series.index.get_level_values('Ticker') if isinstance(cluster_series.index, pd.MultiIndex) else cluster_series.index

        # Get dummies
        H = pd.get_dummies(cluster_series).values # (N_tickers, N_clusters)
        A = H @ H.T

        return pd.DataFrame(A, index=tickers, columns=tickers)
