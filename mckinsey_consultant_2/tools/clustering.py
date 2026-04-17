"""
Clustering and segmentation tools for the Business Analyst agent.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from typing import Dict, List, Any, Tuple
import warnings
warnings.filterwarnings('ignore')


def kmeans_clustering(df: pd.DataFrame, feature_cols: List[str], n_clusters: int = None,
                      auto_select: bool = True) -> Dict[str, Any]:
    """
    Perform K-means clustering with optional automatic cluster selection.
    """
    # Prepare data
    X = df[feature_cols].copy()
    X = X.fillna(X.mean())

    if len(X) < 10:
        return {'error': 'Insufficient data for clustering (need at least 10 samples)'}

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Auto-select number of clusters if requested
    if auto_select or n_clusters is None:
        n_clusters = auto_select_clusters(X_scaled, max_clusters=min(10, len(X) // 10))

    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    # Calculate silhouette score
    silhouette = silhouette_score(X_scaled, labels) if n_clusters > 1 else 0

    # Create cluster summary
    df_copy = df.copy()
    df_copy['cluster'] = labels

    cluster_summaries = {}
    for cluster_id in range(n_clusters):
        cluster_data = df_copy[df_copy['cluster'] == cluster_id]
        cluster_summaries[f'cluster_{cluster_id}'] = {
            'size': len(cluster_data),
            'percentage': len(cluster_data) / len(df) * 100,
            'mean_values': cluster_data[feature_cols].mean().to_dict(),
            'std_values': cluster_data[feature_cols].std().to_dict()
        }

    # Find distinguishing features for each cluster
    cluster_profiles = {}
    for cluster_id in range(n_clusters):
        cluster_data = df_copy[df_copy['cluster'] == cluster_id][feature_cols].mean()
        other_data = df_copy[df_copy['cluster'] != cluster_id][feature_cols].mean()

        # Calculate relative difference
        diff = ((cluster_data - other_data) / other_data * 100).abs().sort_values(ascending=False)
        top_features = diff.head(3).to_dict()

        cluster_profiles[f'cluster_{cluster_id}'] = {
            'distinguishing_features': top_features,
            'size': len(df_copy[df_copy['cluster'] == cluster_id])
        }

    interpretation = f"""
    Data segmented into {n_clusters} distinct clusters using {', '.join(feature_cols)}.
    Silhouette score: {silhouette:.3f} ({'good' if silhouette > 0.5 else 'moderate' if silhouette > 0.25 else 'weak'} separation).
    Largest cluster: Cluster {max(cluster_summaries.items(), key=lambda x: x[1]['size'])[0].replace('cluster_', '')} ({max(cluster_summaries.items(), key=lambda x: x[1]['size'])[1]['percentage']:.1f}% of data).
    """

    return {
        'method': 'kmeans',
        'n_clusters': n_clusters,
        'silhouette_score': silhouette,
        'cluster_labels': labels.tolist(),
        'cluster_summaries': cluster_summaries,
        'cluster_profiles': cluster_profiles,
        'feature_cols': feature_cols,
        'interpretation': interpretation.strip()
    }


def auto_select_clusters(X: np.ndarray, max_clusters: int = 10) -> int:
    """
    Use elbow method and silhouette score to select optimal number of clusters.
    """
    inertias = []
    silhouettes = []
    k_range = range(2, min(max_clusters + 1, len(X) // 5))

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        inertias.append(kmeans.inertia_)
        silhouettes.append(silhouette_score(X, labels))

    # Find elbow point (simplified)
    elbow = 3  # Default
    if len(inertias) > 2:
        # Simple elbow detection: largest second derivative
        diffs = np.diff(inertias)
        if len(diffs) > 1:
            second_diffs = np.diff(diffs)
            if len(second_diffs) > 0:
                elbow_idx = np.argmax(second_diffs) + 2
                elbow = elbow_idx

    # Also consider silhouette score
    best_silhouette_k = list(k_range)[np.argmax(silhouettes)] if silhouettes else 3

    # Take average, preferring lower k
    optimal_k = int(np.round((elbow + best_silhouette_k) / 2))
    return max(2, min(optimal_k, max_clusters))


def rfm_analysis(df: pd.DataFrame, customer_id_col: str, date_col: str,
                 monetary_col: str = None, n_segments: int = 4) -> Dict[str, Any]:
    """
    RFM (Recency, Frequency, Monetary) analysis for customer segmentation.
    Requires transaction data with customer ID and date.
    """
    df_copy = df.copy()

    # Convert date column
    try:
        df_copy[date_col] = pd.to_datetime(df_copy[date_col])
    except:
        return {'error': f'Cannot convert {date_col} to datetime'}

    # Calculate reference date (latest date in dataset + 1 day)
    reference_date = df_copy[date_col].max() + pd.Timedelta(days=1)

    # Calculate RFM metrics
    if monetary_col:
        rfm = df_copy.groupby(customer_id_col).agg({
            date_col: lambda x: (reference_date - x.max()).days,  # Recency
            customer_id_col: 'count',  # Frequency
            monetary_col: 'sum'  # Monetary
        }).reset_index()
        rfm.columns = [customer_id_col, 'recency', 'frequency', 'monetary']
    else:
        rfm = df_copy.groupby(customer_id_col).agg({
            date_col: lambda x: (reference_date - x.max()).days,
            customer_id_col: 'count'
        }).reset_index()
        rfm.columns = [customer_id_col, 'recency', 'frequency']

    # Create RFM scores (1-5 scale, 5 being best)
    rfm['R_score'] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])  # Lower recency = higher score
    rfm['F_score'] = pd.qcut(rfm['frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])

    if monetary_col:
        rfm['M_score'] = pd.qcut(rfm['monetary'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
        rfm['RFM_score'] = rfm['R_score'].astype(str) + rfm['F_score'].astype(str) + rfm['M_score'].astype(str)
    else:
        rfm['RFM_score'] = rfm['R_score'].astype(str) + rfm['F_score'].astype(str)

    # Define customer segments
    def segment_customers(row):
        r, f = int(row['R_score']), int(row['F_score'])
        if r >= 4 and f >= 4:
            return 'Champions'
        elif r >= 3 and f >= 3:
            return 'Loyal Customers'
        elif r >= 4 and f <= 2:
            return 'New Customers'
        elif r >= 3 and f <= 2:
            return 'Potential Loyalists'
        elif r <= 2 and f >= 4:
            return 'At Risk'
        elif r <= 2 and f >= 2:
            return 'Cannot Lose Them'
        elif r <= 2 and f <= 2:
            return 'Lost'
        else:
            return 'Others'

    rfm['segment'] = rfm.apply(segment_customers, axis=1)

    # Segment summary
    segment_summary = rfm.groupby('segment').agg({
        customer_id_col: 'count',
        'recency': 'mean',
        'frequency': 'mean'
    }).rename(columns={customer_id_col: 'count'})

    if monetary_col:
        segment_summary['monetary'] = rfm.groupby('segment')['monetary'].mean()

    segment_summary['percentage'] = segment_summary['count'] / len(rfm) * 100

    # Top segments by value
    if monetary_col:
        top_segments = segment_summary.sort_values('monetary', ascending=False)
    else:
        top_segments = segment_summary.sort_values('frequency', ascending=False)

    interpretation = f"""
    RFM analysis identified {len(segment_summary)} customer segments.
    Top segment: {top_segments.index[0]} ({top_segments.iloc[0]['percentage']:.1f}% of customers)
    """
    if monetary_col:
        interpretation += f" with avg monetary value of ${top_segments.iloc[0]['monetary']:.2f}."

    return {
        'method': 'rfm_analysis',
        'rfm_data': rfm.to_dict('records'),
        'segment_summary': segment_summary.to_dict(),
        'n_customers': len(rfm),
        'top_segment': top_segments.index[0],
        'segments': segment_summary.to_dict(),
        'interpretation': interpretation.strip()
    }


def pareto_analysis(df: pd.DataFrame, group_col: str, value_col: str, top_pct: float = 80) -> Dict[str, Any]:
    """
    Pareto analysis: Find the top segments that contribute to X% of value.
    """
    # Aggregate by group
    group_values = df.groupby(group_col)[value_col].sum().sort_values(ascending=False)

    # Calculate cumulative percentage
    total = group_values.sum()
    cum_pct = (group_values.cumsum() / total * 100)

    # Find how many groups contribute to top_pct of value
    n_top_groups = (cum_pct <= top_pct).sum() + 1
    top_groups = group_values.head(n_top_groups)

    # Statistics
    top_contribution = top_groups.sum() / total * 100
    top_group_percentage = len(top_groups) / len(group_values) * 100

    interpretation = f"""
    Top {len(top_groups)} {group_col}s ({top_group_percentage:.1f}% of all {group_col}s)
    contribute {top_contribution:.1f}% of total {value_col}.
    This is {'a strong' if top_contribution > 70 else 'a moderate' if top_contribution > 50 else 'a weak'} Pareto effect.
    Key segments: {', '.join(top_groups.head(3).index.astype(str))}
    """

    return {
        'method': 'pareto_analysis',
        'top_groups': top_groups.to_dict(),
        'n_top_groups': len(top_groups),
        'total_groups': len(group_values),
        'top_contribution_pct': top_contribution,
        'pareto_ratio': f"{top_group_percentage:.1f}:{100-top_group_percentage:.1f}",
        'interpretation': interpretation.strip()
    }


def cohort_analysis(df: pd.DataFrame, user_id_col: str, date_col: str,
                    action_col: str = None) -> Dict[str, Any]:
    """
    Cohort retention analysis for understanding user retention over time.
    """
    df_copy = df.copy()

    # Convert date
    try:
        df_copy[date_col] = pd.to_datetime(df_copy[date_col])
    except:
        return {'error': f'Cannot convert {date_col} to datetime'}

    # Find first activity date for each user (cohort)
    user_cohorts = df_copy.groupby(user_id_col)[date_col].min().reset_index()
    user_cohorts.columns = [user_id_col, 'cohort_date']
    user_cohorts['cohort_month'] = user_cohorts['cohort_date'].dt.to_period('M')

    # Merge back
    df_copy = df_copy.merge(user_cohorts[[user_id_col, 'cohort_month']], on=user_id_col)

    # Calculate period number for each activity
    df_copy['activity_month'] = df_copy[date_col].dt.to_period('M')
    df_copy['period_number'] = (df_copy['activity_month'] - df_copy['cohort_month']).apply(attrgetter('n'))

    # Create cohort table
    cohort_data = df_copy.groupby(['cohort_month', 'period_number'])[user_id_col].nunique().reset_index()
    cohort_counts = cohort_data.pivot(index='cohort_month', columns='period_number', values=user_id_col)

    # Calculate cohort sizes
    cohort_sizes = user_cohorts.groupby('cohort_month')[user_id_col].nunique()
    cohort_table = cohort_counts.divide(cohort_sizes, axis=0) * 100

    # Calculate average retention by period
    avg_retention = cohort_table.mean(axis=0)

    interpretation = f"""
    Cohort analysis across {len(cohort_sizes)} monthly cohorts.
    Month 0 (initial): {avg_retention.iloc[0]:.1f}% activity.
    Month 1 retention: {avg_retention.iloc[1]:.1f}% if available.
    Average monthly retention: {avg_retention.mean():.1f}%.
    """

    return {
        'method': 'cohort_analysis',
        'cohort_table': cohort_table.to_dict(),
        'cohort_sizes': cohort_sizes.to_dict(),
        'avg_retention': avg_retention.to_dict(),
        'n_cohorts': len(cohort_sizes),
        'interpretation': interpretation.strip()
    }


# Need to import attrgetter for cohort analysis
from operator import attrgetter
