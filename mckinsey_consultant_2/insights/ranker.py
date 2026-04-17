"""
Insight ranking and synthesis for prioritizing findings.
"""

import pandas as pd
from typing import Dict, List, Any, Optional


class InsightRanker:
    """
    Ranks insights by business value and statistical rigor.
    """

    @staticmethod
    def rank_findings(findings: List[Dict[str, Any]], top_n: int = 3) -> List[Dict[str, Any]]:
        """
        Select top N findings based on composite score.
        """
        # Filter out errors
        valid_findings = [f for f in findings if 'error' not in f]

        # Already sorted by business_score from tester
        top_findings = valid_findings[:top_n]

        # Add narrative to each
        for finding in top_findings:
            finding['narrative'] = InsightRanker._generate_narrative(finding)

        return top_findings

    @staticmethod
    def _generate_narrative(finding: Dict[str, Any]) -> str:
        """
        Generate a plain-English narrative for the finding.
        """
        test_type = finding.get('test_type', '')
        interp = finding.get('interpretation', '')

        if test_type == 'pearson_correlation':
            corr = finding.get('correlation', 0)
            p = finding.get('p_value', 1)
            vars = finding.get('variables', [])
            if len(vars) >= 2:
                direction = "positive" if corr > 0 else "negative"
                strength = "strong" if abs(corr) > 0.7 else "moderate" if abs(corr) > 0.3 else "weak"
                return f"{vars[0]} and {vars[1]} show a {strength} {direction} relationship (r={corr:.2f}, p={p:.3f})."

        elif test_type == 't_test':
            g1, g2 = finding.get('group1', ''), finding.get('group2', '')
            var = finding.get('variables', ['', ''])[0]
            diff = finding.get('mean_group1', 0) - finding.get('mean_group2', 0)
            return f"{g1} differs significantly from {g2} in {var} (difference: {diff:.2f})."

        elif test_type == 'trend_analysis':
            direction = finding.get('trend_direction', 'no')
            var = finding.get('variables', ['', ''])[1]
            change = finding.get('percentage_change', 0)
            return f"{var} shows a {direction} trend with {change:.1f}% change over time."

        elif test_type == 'kmeans':
            n_clusters = finding.get('n_clusters', 0)
            return f"Data segments into {n_clusters} distinct clusters with different characteristics."

        elif test_type == 'rfm_analysis':
            top = finding.get('top_segment', '')
            return f"Customer segmentation reveals '{top}' as the highest-value segment."

        return interp


class Synthesizer:
    """
    Synthesizes multiple findings into executive insights.
    """

    @staticmethod
    def synthesize_insights(findings: List[Dict[str, Any]],
                           df: pd.DataFrame,
                           llm_client=None,
                           model: Optional[str] = None) -> Dict[str, Any]:
        """
        Create executive summary from findings.
        """
        selected_model = model if model else "meta-llama/llama-3.2-1b-instruct:free"
        if llm_client:
            return Synthesizer._llm_synthesize(findings, df, llm_client, selected_model)
        else:
            return Synthesizer._template_synthesize(findings, df)

    @staticmethod
    def _template_synthesize(findings: List[Dict[str, Any]],
                                df: pd.DataFrame) -> Dict[str, Any]:
        """
        Template-based synthesis when LLM is unavailable.
        """
        top_findings = findings[:3]

        # Build executive summary
        summary_parts = []
        for i, finding in enumerate(top_findings, 1):
            narrative = finding.get('narrative', finding.get('interpretation', ''))
            summary_parts.append(f"{i}. {narrative}")

        executive_summary = " ".join(summary_parts) if summary_parts else "No significant findings detected."

        # Generate recommendations based on finding types
        recommendations = Synthesizer._generate_recommendations(top_findings)

        # Generate caveats
        caveats = Synthesizer._generate_caveats(df, findings)

        # Build insight cards
        insights = []
        for finding in top_findings:
            insight = {
                'title': Synthesizer._get_insight_title(finding),
                'narrative': finding.get('narrative', ''),
                'test_type': finding.get('test_type', ''),
                'interpretation': finding.get('interpretation', ''),
                'why_it_matters': Synthesizer._get_business_implication(finding),
                'statistics': {
                    'p_value': finding.get('p_value'),
                    'effect_size': finding.get('effect_size'),
                    'business_score': finding.get('business_score', 0)
                },
                'chart_type': Synthesizer._get_chart_type(finding),
                'variables': finding.get('variables', [finding.get('variable')] if finding.get('variable') else []),
                'raw_finding': finding
            }
            insights.append(insight)

        return {
            'executive_summary': executive_summary,
            'top_insights': insights,
            'recommendations': recommendations,
            'caveats': caveats,
            'n_findings': len(findings),
            'n_significant': len([f for f in findings if f.get('significant', False)])
        }

    @staticmethod
    def _generate_recommendations(findings: List[Dict[str, Any]]) -> List[str]:
        """
        Generate actionable recommendations based on findings.
        """
        recommendations = []

        for finding in findings:
            test_type = finding.get('test_type', '')

            if test_type == 'pearson_correlation':
                vars = finding.get('variables', [])
                corr = finding.get('correlation', 0)
                if len(vars) >= 2:
                    if abs(corr) > 0.5:
                        recommendations.append(
                            f"Leverage the relationship between {vars[0]} and {vars[1]} for predictive modeling."
                        )

            elif test_type == 't_test':
                group_col = finding.get('variables', ['', ''])[1]
                recommendations.append(
                    f"Investigate factors causing differences across {group_col} segments."
                )

            elif test_type == 'trend_analysis':
                var = finding.get('variables', ['', ''])[1]
                direction = finding.get('trend_direction', '')
                if direction == 'decreasing':
                    recommendations.append(
                        f"Investigate causes of declining {var} and develop mitigation strategies."
                    )
                elif direction == 'increasing':
                    recommendations.append(
                        f"Capitalize on growing {var} trend through targeted investments."
                    )

            elif test_type == 'kmeans':
                recommendations.append(
                    "Develop segment-specific strategies for each identified customer cluster."
                )

            elif test_type == 'rfm_analysis':
                top = finding.get('top_segment', '')
                recommendations.append(
                    f"Focus retention efforts on '{top}' segment as highest-value customers."
                )

        # Deduplicate
        recommendations = list(dict.fromkeys(recommendations))
        return recommendations[:5]

    @staticmethod
    def _get_business_implication(finding: Dict[str, Any]) -> str:
        """
        Translate a statistical result into business language.
        """
        test_type = finding.get('test_type', '')

        if test_type == 'pearson_correlation':
            vars = finding.get('variables', [])
            corr = finding.get('correlation', 0)
            if len(vars) >= 2:
                direction = "moves in step with" if corr > 0 else "moves opposite to"
                return f"{vars[0]} materially {direction} {vars[1]}, making it a useful operational signal."

        if test_type == 't_test':
            var, group_col = finding.get('variables', ['', ''])
            return f"Performance is not uniform across {group_col}; segment-specific actions on {var} are likely warranted."

        if test_type == 'trend_analysis':
            var = finding.get('variables', ['', ''])[1]
            direction = finding.get('trend_direction', '')
            return f"{var} is shifting {direction} over time, which may indicate momentum that management should either scale or arrest."

        if test_type == 'chi_square':
            vars = finding.get('variables', [])
            if len(vars) >= 2:
                return f"{vars[0]} and {vars[1]} are linked, suggesting decisions in one area should account for the other."

        if 'outlier_detection' in test_type:
            var = finding.get('variable', 'the metric')
            return f"A concentrated tail in {var} may hide operational exceptions, data issues, or premium-value cases worth triage."

        if test_type == 'distribution_analysis':
            var = finding.get('variable', 'the metric')
            return f"The shape of {var} indicates whether planning should rely on averages or more robust distribution-aware targets."

        return "This pattern is material enough to influence prioritization, performance management, or further investigation."

    @staticmethod
    def _generate_caveats(df: pd.DataFrame, findings: List[Dict[str, Any]]) -> List[str]:
        """
        Generate appropriate caveats for the analysis.
        """
        caveats = []

        # Data quality caveats
        missing_pct = (df.isnull().sum() / len(df) * 100).mean()
        if missing_pct > 5:
            caveats.append(f"Dataset has {missing_pct:.1f}% missing values on average, which may affect results.")

        # Sample size caveat
        if len(df) < 100:
            caveats.append(f"Small sample size (n={len(df)}) limits statistical power.")

        # Correlation causation
        has_correlation = any(f.get('test_type') == 'pearson_correlation' for f in findings)
        if has_correlation:
            caveats.append("Correlations do not imply causation. Further investigation needed for causal claims.")

        # Time range
        date_cols = df.select_dtypes(include=['datetime64']).columns
        if len(date_cols) > 0:
            for col in date_cols[:1]:
                date_range = f"{df[col].min()} to {df[col].max()}"
                caveats.append(f"Analysis covers time period: {date_range}. Results may not generalize to other periods.")

        return caveats if caveats else ["No major caveats identified. Standard statistical assumptions apply."]

    @staticmethod
    def _get_insight_title(finding: Dict[str, Any]) -> str:
        """
        Generate a punchy title for an insight.
        """
        test_type = finding.get('test_type', '')
        priority = finding.get('priority', '')

        titles = {
            'pearson_correlation': 'Key Variable Relationship Detected',
            't_test': 'Significant Group Difference Found',
            'chi_square': 'Category Association Discovered',
            'trend_analysis': 'Time Trend Identified',
            'kmeans': 'Natural Data Segments Revealed',
            'rfm_analysis': 'Customer Value Segmentation',
            'outlier_detection': 'Data Anomalies Flagged'
        }

        base_title = titles.get(test_type, 'Significant Finding')

        if priority == 'Critical':
            return f"🔴 {base_title}"
        elif priority == 'High':
            return f"🟡 {base_title}"
        else:
            return f"🟢 {base_title}"

    @staticmethod
    def _get_chart_type(finding: Dict[str, Any]) -> str:
        """
        Determine the best chart type for visualizing the finding.
        """
        test_type = finding.get('test_type', '')

        chart_map = {
            'pearson_correlation': 'scatter',
            't_test': 'bar',
            'chi_square': 'heatmap',
            'trend_analysis': 'line',
            'kmeans': 'scatter',
            'rfm_analysis': 'bar',
            'outlier_detection': 'box',
            'distribution_analysis': 'histogram'
        }

        return chart_map.get(test_type, 'bar')

    @staticmethod
    def _llm_synthesize(findings: List[Dict[str, Any]],
                        df: pd.DataFrame,
                        llm_client,
                        model: str = "meta-llama/llama-3.2-1b-instruct:free") -> Dict[str, Any]:
        """
        Use LLM to generate polished executive summary via OpenRouter.
        """
        # Prepare findings summary for LLM
        findings_text = "\n".join([
            f"Finding {i+1}: {f.get('interpretation', '')} (Business Score: {f.get('business_score', 0):.0f})"
            for i, f in enumerate(findings[:5])
        ])

        prompt = f"""
        As a McKinsey consultant, write an executive summary based on these data analysis findings:

        {findings_text}

        Dataset: {df.shape[0]} rows, {df.shape[1]} columns

        Provide:
        1. A concise executive summary (2-3 sentences)
        2. Three key insights with business implications
        3. Three specific, actionable recommendations
        4. Two caveats about the analysis

        Return ONLY a JSON object with these exact keys: executive_summary, insights, recommendations, caveats
        The insights should be an array of strings. No markdown code blocks, just raw JSON.
        """

        try:
            response = llm_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a McKinsey consultant writing executive summaries."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1500
            )

            content = response.choices[0].message.content

            # Clean up the response
            content = content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            result = json.loads(content)

            # Convert to standard format
            insights = []
            if isinstance(result.get('insights'), list):
                for i, insight_text in enumerate(result['insights']):
                    insights.append({
                        'title': f'Insight {i+1}',
                        'narrative': insight_text if isinstance(insight_text, str) else insight_text.get('narrative', str(insight_text)),
                        'test_type': findings[i].get('test_type', '') if i < len(findings) else '',
                        'interpretation': findings[i].get('interpretation', '') if i < len(findings) else '',
                        'why_it_matters': Synthesizer._get_business_implication(findings[i]) if i < len(findings) else '',
                        'statistics': {},
                        'chart_type': Synthesizer._get_chart_type(findings[i]) if i < len(findings) else 'bar',
                        'variables': findings[i].get('variables', [findings[i].get('variable')] if findings[i].get('variable') else []) if i < len(findings) else [],
                        'raw_finding': findings[i] if i < len(findings) else {}
                    })

            # If no insights parsed, fall back to template
            if not insights:
                return Synthesizer._template_synthesize(findings, df)

            return {
                'executive_summary': result.get('executive_summary', ''),
                'top_insights': insights,
                'recommendations': result.get('recommendations', []),
                'caveats': result.get('caveats', []),
                'n_findings': len(findings),
                'n_significant': len([f for f in findings if f.get('significant', False)])
            }

        except Exception as e:
            print(f"LLM synthesis failed: {e}, falling back to templates")
            # Fall back to template
            return Synthesizer._template_synthesize(findings, df)


import json
