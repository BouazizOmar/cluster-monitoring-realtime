import logging
from typing import Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_explanation(query: str, results: Dict[str, Any], time_period: str = None) -> str:
    """
    Generate a natural language explanation of the query results.
    
    Args:
        query: The original user query
        results: The query results as a dictionary
        time_period: The time period specified in the query (e.g., "last 2 days")
    
    Returns:
        A natural language explanation of the results
    """
    try:
        # Extract key metrics from results
        metrics = results.get('metrics', {})
        if not metrics:
            return "No results found for the specified query."

        # Build the explanation
        explanation_parts = []
        
        # Add time context if available
        if time_period:
            explanation_parts.append(f"For the {time_period}:")
        
        # Add metric explanations
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):
                explanation_parts.append(f"- {metric_name}: {value:.2f}")
            else:
                explanation_parts.append(f"- {metric_name}: {value}")

        # Add trend analysis if available
        trends = results.get('trends', {})
        if trends:
            explanation_parts.append("\nTrend Analysis:")
            for trend_name, trend_value in trends.items():
                if trend_value > 0:
                    explanation_parts.append(f"- {trend_name} shows an upward trend")
                elif trend_value < 0:
                    explanation_parts.append(f"- {trend_name} shows a downward trend")
                else:
                    explanation_parts.append(f"- {trend_name} remains stable")

        # Add any additional insights
        insights = results.get('insights', [])
        if insights:
            explanation_parts.append("\nKey Insights:")
            for insight in insights:
                explanation_parts.append(f"- {insight}")

        # Combine all parts
        explanation = "\n".join(explanation_parts)
        
        logger.info(f"Generated explanation for query: {query}")
        return explanation

    except Exception as e:
        logger.error(f"Error generating explanation: {str(e)}")
        return "Unable to generate explanation due to an error." 