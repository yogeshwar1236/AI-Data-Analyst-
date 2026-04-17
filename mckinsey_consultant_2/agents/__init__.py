"""
CrewAI Agents for the McKinsey-style AI Consulting Team.
Each agent represents a specific consulting role.
"""

from crewai import Agent
from typing import Optional, List, Dict, Any


class ConsultingAgents:
    """
    Factory for creating the consulting team agents.
    """

    @staticmethod
    def create_orchestrator(llm=None) -> Agent:
        """
        Engagement Manager - orchestrates the team and prioritizes insights.
        """
        return Agent(
            role="Engagement Manager",
            goal="Direct the consulting team to uncover the most valuable insights from data efficiently and stop when sufficient value is found.",
            backstory="""You are a former McKinsey partner with 15+ years of experience leading data analytics
            engagements. You know when to stop exploring and when to dig deeper. You prioritize hypotheses
            that drive business value and ensure the team delivers actionable recommendations, not just
            interesting statistics. You are decisive and focused on ROI of analysis time.""",
            verbose=True,
            allow_delegation=True,
            llm=llm,
            tools=[]
        )

    @staticmethod
    def create_explorer(llm=None) -> Agent:
        """
        Data Scientist - runs statistical tests and explores patterns.
        """
        return Agent(
            role="Senior Data Scientist",
            goal="Run automated statistical tests to find correlations, trends, outliers, and patterns without being asked. Identify statistically significant findings.",
            backstory="""You are a PhD-level data scientist with expertise in statistics, econometrics,
            and machine learning. You love uncovering hidden signals in data. You write efficient
            pandas/scipy code and return both numbers and plain-English interpretations. You are
            rigorous about statistical significance and effect sizes. You never present findings
            without p-values or confidence metrics.""",
            verbose=True,
            allow_delegation=False,
            llm=llm,
            tools=[]  # Tools will be injected at task creation
        )

    @staticmethod
    def create_segmenter(llm=None) -> Agent:
        """
        Business Analyst - finds segments, cohorts, and high-value groups.
        """
        return Agent(
            role="Business Analyst",
            goal="Find natural clusters, high-value segments, and behavioral cohorts that matter for business strategy.",
            backstory="""You are a former management consultant specializing in customer segmentation
            and behavioral analysis. You segment customers, products, or transactions to reveal
            where the real business action is. You understand RFM analysis, cohort retention,
            and Pareto principles. You always connect segments to business implications.""",
            verbose=True,
            allow_delegation=False,
            llm=llm,
            tools=[]
        )

    @staticmethod
    def create_visualizer(llm=None) -> Agent:
        """
        Visualization Lead - creates consulting-grade charts.
        """
        return Agent(
            role="Data Visualization Lead",
            goal="Create consulting-grade charts that tell a story - not just show data. Ensure every visual supports an insight.",
            backstory="""You are an expert in data visualization with experience at top consulting
            firms. You know that a waterfall chart explains variance better than a bar chart,
            and that heatmaps reveal patterns tables cannot. You use color, annotations, and
            titles that speak to executives. You believe a chart should be self-explanatory
            within 5 seconds.""",
            verbose=True,
            allow_delegation=False,
            llm=llm,
            tools=[]
        )

    @staticmethod
    def create_reporter(llm=None) -> Agent:
        """
        Report Writer (Associate) - writes polished consulting output.
        """
        return Agent(
            role="Consulting Associate",
            goal="Produce a polished consulting report: executive summary, key findings, recommendations, and caveats.",
            backstory="""You are a top-performing McKinsey associate who writes with clarity and
            precision. You write executive summaries that senior partners can present to CEOs.
            You are evidence-based, concise, and actionable. You never use jargon without
            explanation. You know that a good recommendation has a what, a why, and a how.""",
            verbose=True,
            allow_delegation=False,
            llm=llm,
            tools=[]
        )


# Tool definitions for agents (these will be wrapped as CrewAI tools)
from crewai_tools import BaseTool
from typing TypeVar


class StatisticalTestTool(BaseTool):
    name: str = "Statistical Test Tool"
    description: str = "Run statistical tests on data (correlation, t-test, chi-square, trend, outlier detection)"

    def _run(self, test_type: str, **kwargs) -> Dict[str, Any]:
        from ..tools.stats_tests import (
            correlation_test, group_comparison_test, chi_square_test,
            outlier_detection, trend_test
        )

        test_map = {
            'correlation': correlation_test,
            't_test': group_comparison_test,
            'chi_square': chi_square_test,
            'outlier': outlier_detection,
            'trend': trend_test
        }

        if test_type in test_map:
            return test_map[test_type](**kwargs)
        return {"error": f"Unknown test type: {test_type}"}


class ClusteringTool(BaseTool):
    name: str = "Clustering Tool"
    description: str = "Perform clustering and segmentation analysis (k-means, RFM, cohort analysis)"

    def _run(self, method: str, **kwargs) -> Dict[str, Any]:
        from ..tools.clustering import kmeans_clustering, rfm_analysis, cohort_analysis

        method_map = {
            'kmeans': kmeans_clustering,
            'rfm': rfm_analysis,
            'cohort': cohort_analysis
        }

        if method in method_map:
            return method_map[method](**kwargs)
        return {"error": f"Unknown method: {method}"}


class VisualizationTool(BaseTool):
    name: str = "Visualization Tool"
    description: str = "Create consulting-grade visualizations (waterfall, bar, heatmap, trend, distribution)"

    def _run(self, chart_type: str, **kwargs) -> Dict[str, Any]:
        from ..utils.charts import (
            create_waterfall_chart, create_annotated_bar_chart,
            create_correlation_heatmap, create_trend_chart,
            create_distribution_plot
        )

        chart_map = {
            'waterfall': create_waterfall_chart,
            'bar': create_annotated_bar_chart,
            'heatmap': create_correlation_heatmap,
            'trend': create_trend_chart,
            'distribution': create_distribution_plot
        }

        if chart_type in chart_map:
            return {"chart": chart_map[chart_type](**kwargs)}
        return {"error": f"Unknown chart type: {chart_type}"}
