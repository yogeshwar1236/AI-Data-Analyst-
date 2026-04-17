"""
Statistical testing toolkit for the Data Scientist agent.
Pre-written tests to avoid LLM generating buggy code.
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import pearsonr, spearmanr, ttest_ind, chi2_contingency, f_oneway
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')


def correlation_test(df: pd.DataFrame, col1: str, col2: str, method: str = 'pearson') -> Dict[str, Any]:
    """
    Test correlation between two numeric columns.
    Returns: correlation coefficient, p-value, significance, interpretation
    """
    x = df[col1].dropna()
    y = df[col2].dropna()

    # Align indices
    common_idx = x.index.intersection(y.index)
    x = x.loc[common_idx]
    y = y.loc[common_idx]

    if len(x) < 3:
        return {'error': 'Insufficient data points'}

    if method == 'pearson':
        corr, p_value = pearsonr(x, y)
    else:
        corr, p_value = spearmanr(x, y)

    effect_size = 'large' if abs(corr) > 0.7 else 'medium' if abs(corr) > 0.3 else 'small'
    significant = p_value < 0.05

    interpretation = f""
    if significant:
        interpretation += f"Strong {method} correlation (r={corr:.3f}, p={p_value:.3f})" if abs(corr) > 0.7 else f"Moderate {method} correlation (r={corr:.3f}, p={p_value:.3f})" if abs(corr) > 0.3 else f"Weak {method} correlation (r={corr:.3f}, p={p_value:.3f})"
    else:
        interpretation += f"No significant correlation (r={corr:.3f}, p={p_value:.3f})"

    return {
        'test_type': f'{method}_correlation',
        'correlation': corr,
        'p_value': p_value,
        'significant': significant,
        'effect_size': effect_size,
        'n_samples': len(x),
        'interpretation': interpretation,
        'variables': [col1, col2]
    }


def group_comparison_test(df: pd.DataFrame, numeric_col: str, group_col: str,
                          group1: Optional[str] = None, group2: Optional[str] = None) -> Dict[str, Any]:
    """
    Compare means between two groups using t-test.
    If group1/group2 not specified, compares first two unique values.
    """
    groups = df[group_col].dropna().unique()

    if len(groups) < 2:
        return {'error': f'Need at least 2 groups in {group_col}, found {len(groups)}'}

    if group1 is None:
        group1 = groups[0]
    if group2 is None:
        group2 = groups[1]

    values1 = df[df[group_col] == group1][numeric_col].dropna()
    values2 = df[df[group_col] == group2][numeric_col].dropna()

    if len(values1) < 3 or len(values2) < 3:
        return {'error': 'Insufficient data in one or both groups'}

    t_stat, p_value = ttest_ind(values1, values2)

    mean1, mean2 = values1.mean(), values2.mean()
    pooled_std = np.sqrt(((len(values1) - 1) * values1.var() + (len(values2) - 1) * values2.var()) /
                         (len(values1) + len(values2) - 2))
    cohen_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0

    effect_size = 'large' if abs(cohen_d) > 0.8 else 'medium' if abs(cohen_d) > 0.5 else 'small'
    significant = p_value < 0.05

    interpretation = f""
    if significant:
        if mean1 > mean2:
            interpretation += f"{group1} has significantly higher {numeric_col} than {group2} (difference: {mean1-mean2:.2f}, p={p_value:.3f})"
        else:
            interpretation += f"{group2} has significantly higher {numeric_col} than {group1} (difference: {mean2-mean1:.2f}, p={p_value:.3f})"
    else:
        interpretation += f"No significant difference in {numeric_col} between {group1} and {group2} (p={p_value:.3f})"

    return {
        'test_type': 't_test',
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': significant,
        'effect_size': effect_size,
        'cohen_d': cohen_d,
        'mean_group1': mean1,
        'mean_group2': mean2,
        'group1': group1,
        'group2': group2,
        'n_group1': len(values1),
        'n_group2': len(values2),
        'interpretation': interpretation,
        'variables': [numeric_col, group_col]
    }


def chi_square_test(df: pd.DataFrame, col1: str, col2: str) -> Dict[str, Any]:
    """
    Test independence between two categorical columns using chi-square test.
    """
    contingency = pd.crosstab(df[col1], df[col2])

    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
        return {'error': 'Need at least 2 categories in each variable'}

    chi2, p_value, dof, expected = chi2_contingency(contingency)

    # Cramer's V for effect size
    n = contingency.sum().sum()
    cramers_v = np.sqrt(chi2 / (n * (min(contingency.shape) - 1)))

    effect_size = 'large' if cramers_v > 0.5 else 'medium' if cramers_v > 0.3 else 'small'
    significant = p_value < 0.05

    interpretation = f""
    if significant:
        interpretation += f"Significant association between {col1} and {col2} (chi2={chi2:.2f}, p={p_value:.3f})"
    else:
        interpretation += f"No significant association between {col1} and {col2} (p={p_value:.3f})"

    return {
        'test_type': 'chi_square',
        'chi2': chi2,
        'p_value': p_value,
        'dof': dof,
        'significant': significant,
        'effect_size': effect_size,
        'cramers_v': cramers_v,
        'interpretation': interpretation,
        'variables': [col1, col2]
    }


def outlier_detection(df: pd.DataFrame, col: str, method: str = 'iqr') -> Dict[str, Any]:
    """
    Detect outliers using IQR or z-score method.
    """
    data = df[col].dropna()

    if len(data) == 0:
        return {'error': f'No valid data in column {col}'}

    if method == 'iqr':
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = data[(data < lower_bound) | (data > upper_bound)]
    else:  # z-score
        z_scores = np.abs(stats.zscore(data))
        outliers = data[z_scores > 3]

    outlier_pct = len(outliers) / len(data) * 100

    interpretation = f""
    if outlier_pct > 5:
        interpretation += f"High outlier rate ({outlier_pct:.1f}%) in {col} - may indicate data quality issues or natural extreme values"
    elif outlier_pct > 1:
        interpretation += f"Moderate outlier rate ({outlier_pct:.1f}%) in {col}"
    else:
        interpretation += f"Low outlier rate ({outlier_pct:.1f}%) in {col} - data appears clean"

    return {
        'test_type': f'{method}_outlier_detection',
        'outlier_count': len(outliers),
        'outlier_percentage': outlier_pct,
        'outlier_values': outliers.tolist()[:10],  # First 10 outliers
        'method': method,
        'interpretation': interpretation,
        'variable': col
    }


def trend_test(df: pd.DataFrame, time_col: str, value_col: str) -> Dict[str, Any]:
    """
    Test for trend in time series data using Spearman correlation.
    """
    df_copy = df.copy()

    # Try to convert to datetime
    try:
        df_copy[time_col] = pd.to_datetime(df_copy[time_col], errors='coerce')
        df_copy[value_col] = pd.to_numeric(df_copy[value_col], errors='coerce')
        df_copy = df_copy.dropna(subset=[time_col, value_col]).sort_values(time_col)
    except Exception:
        return {'error': f'Cannot convert {time_col} to datetime'}

    # Create time index (numeric)
    time_index = np.arange(len(df_copy))
    values = df_copy[value_col].to_numpy(dtype=float)

    if len(values) < 10:
        return {'error': 'Insufficient time points for trend analysis'}

    corr, p_value = spearmanr(time_index, values)

    trend_direction = 'increasing' if corr > 0 else 'decreasing'
    significant = p_value < 0.05

    # Calculate percentage change
    first_val = values[:len(values)//5].mean()  # First 20%
    last_val = values[-len(values)//5:].mean()   # Last 20%
    pct_change = ((last_val - first_val) / first_val * 100) if first_val != 0 else 0

    interpretation = f""
    if significant:
        interpretation += f"Significant {trend_direction} trend in {value_col} over time (correlation={corr:.3f}, p={p_value:.3f}). Total change: {pct_change:.1f}%"
    else:
        interpretation += f"No significant trend detected in {value_col} (p={p_value:.3f})"

    return {
        'test_type': 'trend_analysis',
        'correlation': corr,
        'p_value': p_value,
        'significant': significant,
        'trend_direction': trend_direction if significant else 'none',
        'percentage_change': pct_change,
        'n_points': len(values),
        'interpretation': interpretation,
        'variables': [time_col, value_col]
    }


def regression_analysis(df: pd.DataFrame, target_col: str, feature_cols: List[str]) -> Dict[str, Any]:
    """
    Simple linear regression analysis.
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score

    X = df[feature_cols].fillna(df[feature_cols].mean())
    y = df[target_col].fillna(df[target_col].mean())

    # Align indices
    valid_idx = X.dropna().index.intersection(y.dropna().index)
    X = X.loc[valid_idx]
    y = y.loc[valid_idx]

    if len(X) < 10:
        return {'error': 'Insufficient samples for regression'}

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)

    # Feature importance (coefficients)
    feature_importance = dict(zip(feature_cols, model.coef_))
    sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)

    interpretation = f""
    interpretation += f"Regression model explains {r2*100:.1f}% of variance in {target_col}. "
    interpretation += f"Top predictor: {sorted_features[0][0]} (coefficient: {sorted_features[0][1]:.3f})"

    return {
        'test_type': 'linear_regression',
        'r_squared': r2,
        'adj_r_squared': 1 - (1 - r2) * (len(y) - 1) / (len(y) - X.shape[1] - 1),
        'feature_importance': feature_importance,
        'top_predictor': sorted_features[0][0] if sorted_features else None,
        'intercept': model.intercept_,
        'interpretation': interpretation,
        'target': target_col,
        'features': feature_cols
    }


def distribution_test(df: pd.DataFrame, col: str) -> Dict[str, Any]:
    """
    Analyze distribution characteristics of a column.
    """
    data = df[col].dropna()

    if len(data) < 10:
        return {'error': 'Insufficient data for distribution analysis'}

    mean = data.mean()
    median = data.median()
    std = data.std()
    skewness = data.skew()
    kurt = data.kurtosis()

    # Normality test
    stat, p_value = stats.normaltest(data)
    is_normal = p_value > 0.05

    distribution_type = 'normal' if is_normal else 'right-skewed' if skewness > 0.5 else 'left-skewed' if skewness < -0.5 else 'approximately symmetric'

    interpretation = f""
    interpretation += f"{col} shows {distribution_type} distribution (mean={mean:.2f}, median={median:.2f}, std={std:.2f}). "
    if abs(skewness) > 1:
        interpretation += f"High skewness ({skewness:.2f}) suggests using median for central tendency."
    else:
        interpretation += f"Skewness ({skewness:.2f}) is within normal range."

    return {
        'test_type': 'distribution_analysis',
        'mean': mean,
        'median': median,
        'std': std,
        'skewness': skewness,
        'kurtosis': kurt,
        'is_normal': is_normal,
        'distribution_type': distribution_type,
        'interpretation': interpretation,
        'variable': col
    }


def auto_explore(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Automatically run all relevant tests on a dataframe.
    Returns list of findings.
    """
    findings = []

    # Get column types
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Try to find date column
    date_col = None
    for col in df.columns:
        series = df[col]

        if pd.api.types.is_datetime64_any_dtype(series):
            date_col = col
            break

        if pd.api.types.is_numeric_dtype(series):
            continue

        parsed_dates = pd.to_datetime(series, errors='coerce')
        valid_ratio = parsed_dates.notna().mean()
        if valid_ratio >= 0.8 and parsed_dates.nunique() > 1:
            date_col = col
            break

    if date_col in numeric_cols:
        numeric_cols.remove(date_col)

    # Run tests

    # 1. Distribution analysis for all numeric columns
    for col in numeric_cols[:5]:  # Limit to first 5
        result = distribution_test(df, col)
        if 'error' not in result:
            findings.append(result)

    # 2. Correlation analysis (top pairs)
    if len(numeric_cols) >= 2:
        corr_matrix = df[numeric_cols].corr().abs()
        # Get top correlations (excluding diagonal)
        corr_pairs = []
        for i in range(len(numeric_cols)):
            for j in range(i+1, len(numeric_cols)):
                corr_pairs.append((numeric_cols[i], numeric_cols[j], corr_matrix.iloc[i, j]))
        corr_pairs.sort(key=lambda x: x[2], reverse=True)

        for col1, col2, corr_val in corr_pairs[:3]:
            result = correlation_test(df, col1, col2)
            if 'error' not in result:
                findings.append(result)

    # 3. Outlier detection
    for col in numeric_cols[:3]:
        result = outlier_detection(df, col)
        if 'error' not in result:
            findings.append(result)

    # 4. Group comparisons (if categorical columns exist)
    if categorical_cols and numeric_cols:
        cat_col = categorical_cols[0]
        num_col = numeric_cols[0]
        result = group_comparison_test(df, num_col, cat_col)
        if 'error' not in result:
            findings.append(result)

    # 5. Trend analysis (if date column found)
    if date_col and numeric_cols:
        for num_col in numeric_cols[:2]:
            result = trend_test(df, date_col, num_col)
            if 'error' not in result:
                findings.append(result)

    # 6. Chi-square for categorical pairs
    if len(categorical_cols) >= 2:
        result = chi_square_test(df, categorical_cols[0], categorical_cols[1])
        if 'error' not in result:
            findings.append(result)

    return findings
