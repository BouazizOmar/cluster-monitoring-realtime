import os
from openai import OpenAI
from dotenv import load_dotenv
import pandas as pd
import re

from schema import get_schema_metadata
from sql_generation import generate_safe_sql, clean_sql_query
from query_executor import execute_query

load_dotenv()

class SnowflakeAIAgent:
    def __init__(self):
        # Simple initialization compatible with newer OpenAI versions
        try:
            self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        except TypeError as e:
            # Fall back to an alternative initialization if needed
            import httpx
            self.openai_client = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                http_client=httpx.Client(timeout=60.0)
            )
            
        self.snowflake_config = {
            "user": os.getenv("SNOWFLAKE_USER"),
            "password": os.getenv("SNOWFLAKE_PASSWORD"),
            "account": os.getenv("SNOWFLAKE_ACCOUNT"),
            "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE"),
            "database": os.getenv("SNOWFLAKE_DATABASE"),
            "schema": os.getenv("SNOWFLAKE_SCHEMA"),
            "role": os.getenv("SNOWFLAKE_ROLE", "READ_ONLY_ROLE"),
        }
        self.schema_metadata = get_schema_metadata()
    
    def process_question(self, user_question: str):
        """
        Process a natural language question and return SQL query results
        
        Args:
            user_question: The natural language question to process
            
        Returns:
            A dictionary containing the query, results, and explanation
        """
        try:
            # Generate SQL query (returns a tuple with main query and debug queries)
            query_result = generate_safe_sql(
                user_query=user_question,
                schema_metadata=self.schema_metadata,
                openai_client=self.openai_client
            )
            
            # Extract main query (first item) and debug queries (second item)
            if isinstance(query_result, tuple) and len(query_result) == 2:
                main_query, debug_queries = query_result
            else:
                main_query = query_result
                debug_queries = []
            
            # Clean the main query to remove comments and extra whitespace
            cleaned_query = clean_sql_query(main_query)
            
            # Execute the main query
            results_df = execute_query(cleaned_query, self.snowflake_config)
            
            # If the query returns no results, try to get diagnostic information
            if results_df is None or results_df.empty:
                # Get available VM names for reference
                vm_query = """
                SELECT DISTINCT VM, VM_KEY FROM DIM_VM ORDER BY VM
                """
                vm_df = execute_query(vm_query, self.snowflake_config, is_debug=True)
                
                # Get available metrics that might be related to the query
                if 'memory' in user_question.lower():
                    metrics_filter = "METRIC_NAME LIKE '%memory%' OR METRIC_NAME LIKE '%mem%'"
                elif 'cpu' in user_question.lower():
                    metrics_filter = "METRIC_NAME LIKE '%cpu%'"
                else:
                    metrics_filter = "1=1"
                    
                metric_query = f"""
                SELECT DISTINCT METRIC_KEY, METRIC_NAME 
                FROM DIM_METRIC 
                WHERE {metrics_filter}
                ORDER BY METRIC_NAME
                """
                metric_df = execute_query(metric_query, self.snowflake_config, is_debug=True)
                
                # Add diagnostic information to the explanation
                diagnostic_info = {
                    "vm_names": vm_df['VM'].tolist() if vm_df is not None and not vm_df.empty else [],
                    "metrics": metric_df['METRIC_NAME'].tolist() if metric_df is not None and not metric_df.empty else []
                }
            else:
                diagnostic_info = None
            
            # Convert DataFrame to a cleaner format for display
            results_dict = self._format_results(results_df)
            
            # Generate explanation using OpenAI
            explanation = self._explain_results(cleaned_query, results_df, user_question, diagnostic_info)
            
            return {
                "success": True,
                "query": cleaned_query,  # Return only the main query
                "results": results_dict,
                "explanation": explanation
            }
        except Exception as e:
            import traceback
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    def _format_results(self, df):
        """
        Format DataFrame results into a clean dictionary format
        
        Args:
            df: Pandas DataFrame with query results
            
        Returns:
            Dictionary representation of results
        """
        if df is None or df.empty:
            return {"rows": [], "columns": []}
        
        import numpy as np
        import math
        import json
        
        # Handle problematic float values (NaN, Infinity) before JSON serialization
        # First, make a copy to avoid modifying the original DataFrame
        df_copy = df.copy()
        
        # Define a safe converter function
        def json_safe_value(val):
            if val is None:
                return None
            # Handle NaN, Infinity in all their forms
            if isinstance(val, (float, np.float64, np.float32, np.float16)):
                if math.isnan(val) or np.isnan(val):
                    return None
                if math.isinf(val) or np.isinf(val):
                    return None
            # Handle Timestamp objects
            if hasattr(val, 'isoformat'):
                return val.isoformat()
            # Handle NumPy types
            if isinstance(val, (np.int64, np.int32, np.int16, np.int8)):
                return int(val)
            # Handle string 'NaN'
            if isinstance(val, str) and val == 'NaN':
                return None
            return val
        
        # Process each column
        for col in df_copy.columns:
            df_copy[col] = df_copy[col].apply(json_safe_value)
        
        try:
            # First convert to records
            records = df_copy.to_dict(orient="records")
            
            # Manually serialize to verify it works
            json.dumps(records)
            
            return {
                "rows": records,
                "columns": df_copy.columns.tolist(),
                "row_count": len(df_copy)
            }
        except Exception as e:
            # If there's still a serialization error, use an aggressive string conversion
            print(f"Warning: JSON serialization error: {str(e)}")
            string_df = df_copy.applymap(lambda x: str(x) if x is not None else None)
            
            return {
                "rows": string_df.to_dict(orient="records"),
                "columns": string_df.columns.tolist(),
                "row_count": len(string_df)
            }
    
    def _explain_results(self, query, results_df, user_question, diagnostic_info=None):
        """
        Generate a natural language explanation of the query results
        
        Args:
            query: The SQL query executed
            results_df: DataFrame with query results
            user_question: Original user question
            diagnostic_info: Dictionary containing diagnostic information
            
        Returns:
            A natural language explanation of the results
        """
        try:
            if results_df is not None and not results_df.empty:
                sample_df = results_df.head(10) if len(results_df) > 10 else results_df
                result_str = sample_df.to_string()
                row_count = len(results_df)
            else:
                # If we have no results, add diagnostic information
                result_str = "No results or empty dataset"
                row_count = 0
                
                # Include diagnostic info if available
                if diagnostic_info:
                    vm_list = diagnostic_info.get("vm_names", [])
                    metric_list = diagnostic_info.get("metrics", [])
                    
                    # Get metric type based on query
                    if 'memory' in user_question.lower():
                        metric_type = 'memory'
                    elif 'cpu' in user_question.lower():
                        metric_type = 'CPU'
                    else:
                        metric_type = 'relevant'
                    
                    # Add diagnostic information
                    result_str += f"\n\nDiagnostic Information:\n- Available VM names: {', '.join(vm_list) if vm_list else 'None found'}\n- Available {metric_type} metrics: {', '.join(metric_list) if metric_list else 'None found'}"
            
            prompt = f'''
            The user asked: "{user_question}"
            
            The following SQL query was executed:
            ```sql
            {query}
            ```
            
            The query returned {row_count} total rows. Here's a sample of the results:
            ```
            {result_str}
            ```
            
            Please provide a concise, clear explanation of these results in relation to the user's question.
            Focus on the most important insights from the data.
            
            If the query returned zero results, carefully explain possible reasons including:
            1. The specific filter values or VM names might not match exactly what's in the database
            2. The time period examined might not contain any relevant data
            3. The memory metrics might be missing for certain VMs
            4. There may be a need to join the data differently to see the correct metrics
            
            Provide actionable suggestions on how to modify the query for better results.
            
            DO NOT include markdown formatting in your response.
            DO NOT use numbered lists or bullet points in your response.
            DO NOT suggest follow-up questions.
            Respond in a few short paragraphs only.
            '''
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a data analyst explaining query results clearly and helpfully."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"Error generating explanation: {str(e)}"