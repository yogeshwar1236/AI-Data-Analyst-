"""
Hypothesis testing engine - maps hypotheses to statistical tests.
"""

import pandas as pd
from typing import Dict, List, Any
import sys
import os

# Add tools to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.stats_tests import (
    correlation_test, group_comparison_test, chi_square_test,
    outlier_detection, trend_test, regression_analysis, distribution_test
)
from tools.clustering import kmeans_clustering


class HypothesisTester:
    """
    Tests hypotheses using appropriate statistical methods.
    Returns results with statistical significance and business impact.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def test_hypothesis(self, hypothesis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Route hypothesis to appropriate test method.
        """
        test_method = hypothesis.get('test_method', '')
        variables = hypothesis.get('variables', [])

        try:
            if test_method == 'pearson_correlation':
                if len(variables) >= 2:
                    result = correlation_test(self.df, variables[0], variables[1], method='pearson')
                else:
                    return {'error': 'Insufficient variables for correlation test'}

            elif test_method == 't_test':
                if len(variables) >= 2:
                    result = group_comparison_test(self.df, variables[0], variables[1])
                else:
                    return {'error': 'Insufficient variables for t-test'}

            elif test_method == 'chi_square':
                if len(variables) >= 2:
                    result = chi_square_test(self.df, variables[0], variables[1])
                else:
                    return {'error': 'Insufficient variables for chi-square test'}

            elif test_method == 'trend_analysis':
                if len(variables) >= 2:
                    result = trend_test(self.df, variables[0], variables[1])
                else:
                    return {'error': 'Insufficient variables for trend analysis'}

            elif test_method == 'outlier_detection':
                if len(variables) >= 1:
                    result = outlier_detection(self.df, variables[0])
                else:
                    return {'error': 'No variable specified for outlier detection'}

            elif test_method == 'distribution_analysis':
                if len(variables) >= 1:
                    result = distribution_test(self.df, variables[0])
                else:
                    return {'error': 'No variable specified for distribution test'}

            elif test_method == 'kmeans_clustering':
                numeric_vars = [v for v in variables if v in self.df.select_dtypes(include=[float, int]).columns]
                if len(numeric_vars) >= 2:
                    result = kmeans_clustering(self.df, numeric_vars)
                else:
                    return {'error': 'Insufficient numeric variables for clustering'}

            elif test_method == 'linear_regression':
                if len(variables) >= 2:
                    result = regression_analysis(self.df, variables[0], variables[1:])
                else:
                    return {'error': 'Insufficient variables for regression'}

            else:
                # Auto-detect based on hypothesis type
                result = self._auto_test(hypothesis)

        except Exception as e:
            result = {'error': str(e), 'hypothesis': hypothesis['statement']}

        # Enrich result with business impact score
        result = self._calculate_business_impact(result, hypothesis)
        result['original_hypothesis'] = hypothesis

        return result

    def _auto_test(self, hypothesis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Automatically select and run appropriate test based on hypothesis type.
        """
        h_type = hypothesis.get('type', '')
        variables = hypothesis.get('variables', [])

        if h_type == 'correlation' and len(variables) >= 2:
            return correlation_test(self.df, variables[0], variables[1])

        elif h_type == 'difference' and len(variables) >= 2:
            return group_comparison_test(self.df, variables[0], variables[1])

        elif h_type == 'trend' and len(variables) >= 2:
            return trend_test(self.df, variables[0], variables[1])

        elif h_type == 'association' and len(variables) >= 2:
            return chi_square_test(self.df, variables[0], variables[1])

        elif h_type == 'distribution' and len(variables) >= 1:
            return distribution_test(self.df, variables[0])

        elif h_type == 'cluster':
            numeric_vars = [v for v in variables if v in self.df.select_dtypes(include=[float, int]).columns]
            if len(numeric_vars) >= 2:
                return kmeans_clustering(self.df, numeric_vars)

        return {'error': f'Cannot auto-test hypothesis type: {h_type}'}

    def _calculate_business_impact(self, result: Dict[str, Any],
                                    hypothesis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate a business impact score for the finding.
        Combines statistical significance with effect size and business relevance.
        """
        if 'error' in result:
            result['business_score'] = 0
            return result

        # Base score from statistical significance
        if result.get('significant', False):
            base_score = 50
        elif 'p_value' in result:
            base_score = max(0, 50 - result['p_value'] * 100)
        else:
            base_score = 25

        # Add effect size component
        effect_score = 0
        if 'correlation' in result:
            effect_score = abs(result['correlation']) * 30
        elif 'cohen_d' in result:
            effect_score = min(abs(result['cohen_d']) * 15, 30)
        elif 'r_squared' in result:
            effect_score = result['r_squared'] * 30
        elif 'chi2' in result:
            effect_score = min(result.get('cramers_v', 0) * 30, 30)

        # Priority bonus
        priority_bonus = {'high': 10, 'medium': 5, 'low': 0}.get(
            hypothesis.get('priority', 'medium'), 0
        )

        result['business_score'] = min(base_score + effect_score + priority_bonus, 100)

        # Priority ranking
        if result['business_score'] >= 80:
            result['priority'] = 'Critical'
        elif result['business_score'] >= 60:
            result['priority'] = 'High'
        elif result['business_score'] >= 40:
            result['priority'] = 'Medium'
        else:
            result['priority'] = 'Low'

        return result

    def test_all(self, hypotheses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Test all hypotheses and return sorted results.
        """
        results = []
        for hypo in hypotheses:
            result = self.test_hypothesis(hypo)
            results.append(result)

        # Sort by business score descending
        results.sort(key=lambda x: x.get('business_score', 0), reverse=True)
        return results
