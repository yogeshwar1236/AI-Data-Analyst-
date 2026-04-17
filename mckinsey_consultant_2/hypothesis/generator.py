"""
Hypothesis generation using LLM to create testable business hypotheses.
"""

import json
from typing import Dict, List, Any, Optional
import os


class HypothesisGenerator:
    """
    Generates testable business hypotheses from data profiles.
    Uses a cheap LLM to generate hypotheses without heavy computation.
    """

    def __init__(self, llm_client=None, model: Optional[str] = None):
        self.llm_client = llm_client
        self.model = model if model else "meta-llama/llama-3.2-1b-instruct:free"  # Default free model

    def generate_from_profile(self, profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate hypotheses based on data profile.
        Falls back to template-based generation if no LLM available.
        """
        if self.llm_client:
            return self._llm_generate(profile)
        else:
            return self._template_generate(profile)

    def _llm_generate(self, profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Use LLM (via OpenRouter or direct) to generate creative, business-focused hypotheses.
        """
        prompt = f"""
        You are a McKinsey data scientist. Given this dataset profile, generate 5-7 specific,
        testable business hypotheses in the format "If [condition], then [outcome] because [reason]".

        Dataset Profile:
        - Columns: {profile.get('columns', [])}
        - Types: {profile.get('dtypes', {})}
        - Numeric columns: {profile.get('numeric_cols', [])}
        - Categorical columns: {profile.get('categorical_cols', [])}
        - Time column: {profile.get('time_col', 'None')}
        - Shape: {profile.get('shape', [0, 0])}
        - Sample values: {profile.get('samples', {})}

        For each hypothesis, specify:
        1. type: one of [correlation, difference, trend, association, cluster, distribution]
        2. variables: list of columns involved
        3. priority: high/medium/low based on business impact potential
        4. test_method: the statistical test to use

        Return ONLY a JSON array of hypotheses. No markdown, no explanation.
        """

        try:
            # OpenRouter uses OpenAI-compatible chat completions
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a data science expert that generates testable business hypotheses."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1500
            )

            content = response.choices[0].message.content

            # Clean up the response - remove markdown code blocks if present
            content = content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            hypotheses = json.loads(content)

            # Validate and clean up hypotheses
            validated_hypotheses = []
            for h in hypotheses[:7]:
                if isinstance(h, dict) and 'statement' in h:
                    validated_hypotheses.append({
                        'statement': h.get('statement', ''),
                        'type': h.get('type', 'correlation'),
                        'variables': h.get('variables', []),
                        'priority': h.get('priority', 'medium'),
                        'test_method': h.get('test_method', 'correlation')
                    })

            return validated_hypotheses if validated_hypotheses else self._template_generate(profile)

        except Exception as e:
            print(f"LLM generation failed: {e}, falling back to templates")
            return self._template_generate(profile)

    def _template_generate(self, profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate hypotheses using templates when LLM is unavailable.
        """
        hypotheses = []
        numeric = profile.get('numeric_cols', [])
        categorical = profile.get('categorical_cols', [])
        time_col = profile.get('time_col')

        # 1. Correlation hypotheses (top variable pairs)
        if len(numeric) >= 2:
            for i in range(min(3, len(numeric))):
                for j in range(i+1, min(i+3, len(numeric))):
                    hypotheses.append({
                        'statement': f"If {numeric[i]} increases, {numeric[j]} tends to increase due to potential causal relationship",
                        'type': 'correlation',
                        'variables': [numeric[i], numeric[j]],
                        'priority': 'high' if i < 1 else 'medium',
                        'test_method': 'pearson_correlation'
                    })

        # 2. Group difference hypotheses
        if categorical and numeric:
            for num in numeric[:3]:
                for cat in categorical[:2]:
                    groups = profile.get('unique_values', {}).get(cat, [])
                    if len(groups) >= 2:
                        hypotheses.append({
                            'statement': f"Different values of {cat} lead to different average {num} due to segment-specific behavior",
                            'type': 'difference',
                            'variables': [num, cat],
                            'priority': 'high',
                            'test_method': 't_test'
                        })

        # 3. Trend hypotheses
        if time_col and numeric:
            for num in numeric[:3]:
                hypotheses.append({
                    'statement': f"{num} shows a significant trend over time due to underlying market or operational changes",
                    'type': 'trend',
                    'variables': [time_col, num],
                    'priority': 'high',
                    'test_method': 'trend_analysis'
                    })

        # 4. Association hypotheses
        if len(categorical) >= 2:
            for i in range(min(2, len(categorical))):
                for j in range(i+1, min(i+2, len(categorical))):
                    hypotheses.append({
                        'statement': f"{categorical[i]} and {categorical[j]} are associated due to shared underlying factors",
                        'type': 'association',
                        'variables': [categorical[i], categorical[j]],
                        'priority': 'medium',
                        'test_method': 'chi_square'
                    })

        # 5. Distribution/outlier hypotheses
        if numeric:
            for num in numeric[:2]:
                hypotheses.append({
                    'statement': f"{num} contains significant outliers that may indicate data quality issues or special cases",
                    'type': 'distribution',
                    'variables': [num],
                    'priority': 'medium',
                    'test_method': 'outlier_detection'
                })

        # 6. Clustering hypothesis
        if len(numeric) >= 2:
            hypotheses.append({
                'statement': f"Data naturally segments into distinct groups based on {', '.join(numeric[:3])}",
                'type': 'cluster',
                'variables': numeric[:4],
                'priority': 'medium',
                'test_method': 'kmeans_clustering'
            })

        return hypotheses[:7]


def profile_data(df) -> Dict[str, Any]:
    """
    Generate a profile of the data for hypothesis generation.
    """
    import pandas as pd
    import numpy as np

    profile = {
        'columns': df.columns.tolist(),
        'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
        'shape': df.shape,
        'numeric_cols': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_cols': df.select_dtypes(include=['object', 'category']).columns.tolist(),
    }

    # Try to find time column
    time_col = None
    for col in df.columns:
        try:
            pd.to_datetime(df[col])
            time_col = col
            break
        except:
            continue
    profile['time_col'] = time_col

    # Sample values for context
    samples = {}
    for col in df.columns[:5]:
        if col in profile['numeric_cols']:
            samples[col] = {
                'min': df[col].min(),
                'max': df[col].max(),
                'mean': df[col].mean()
            }
        else:
            samples[col] = df[col].value_counts().head(3).to_dict()
    profile['samples'] = samples

    # Unique values for categorical
    unique_values = {}
    for col in profile['categorical_cols'][:5]:
        unique_values[col] = df[col].dropna().unique().tolist()[:10]
    profile['unique_values'] = unique_values

    return profile
