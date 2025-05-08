import logging
from typing import Dict, Any, Optional, List, Tuple
from .sql_generation import generate_safe_sql
from .query_executor import execute_query, execute_debug_queries
from .explanation_generator import generate_explanation

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SnowflakeAIAgent:
    def __init__(self, snowflake_config: Dict[str, str]):
        """
        Initialize the Snowflake AI Agent.
        
        Args:
            snowflake_config: Dictionary containing Snowflake connection parameters
        """
        self.snowflake_config = snowflake_config
        logger.info("SnowflakeAIAgent initialized")

    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a natural language query and return results with explanation.
        
        Args:
            query: Natural language query from the user
            
        Returns:
            Dictionary containing query results and explanation
        """
        try:
            # Generate SQL query and get debug queries
            logger.info(f"Processing query: {query}")
            sql_query, debug_queries = generate_safe_sql(query)
            if not sql_query:
                return {"error": "Failed to generate SQL query"}

            # Execute debug queries first
            if debug_queries:
                logger.info("Executing debug queries...")
                debug_results = execute_debug_queries(debug_queries, self.snowflake_config)
                for i, result in enumerate(debug_results, 1):
                    logger.info(f"\nDebug Query {i} Results:")
                    logger.info(result)

            # Execute main query
            results_df = execute_query(sql_query, self.snowflake_config)
            if results_df is None:
                return {"error": "Failed to execute query"}

            # Extract time period from query
            time_period = None
            if "last" in query.lower():
                time_period = query.split("last")[-1].strip()

            # Generate explanation
            explanation = generate_explanation(query, results_df.to_dict(), time_period)

            return {
                "sql_query": sql_query[0] if isinstance(sql_query, tuple) else sql_query,
                "results": results_df.to_dict(),
                "explanation": explanation
            }

        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {"error": f"An error occurred: {str(e)}"}

    def get_metrics(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Get metrics for a specific query.
        
        Args:
            query: Natural language query
            
        Returns:
            Dictionary containing metrics or None if error
        """
        try:
            result = self.process_query(query)
            if "error" in result:
                return None
            return result.get("results", {})
        except Exception as e:
            logger.error(f"Error getting metrics: {str(e)}")
            return None 