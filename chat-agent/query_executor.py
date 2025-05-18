"""
Query executor for Snowflake database
"""
import pandas as pd
import numpy as np
import snowflake.connector
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def execute_query(sql_query, snowflake_config, is_debug=False):
    """
    Execute a SQL query and return results as a pandas DataFrame
    
    Args:
        sql_query: The SQL query to execute
        snowflake_config: Snowflake connection configuration
        is_debug: Whether this is a debug query
        
    Returns:
        pandas DataFrame with query results
    """
    try:
        # Extract the query if it's in a tuple
        if isinstance(sql_query, tuple):
            sql_query = sql_query[0]
        
        # Make sure no markdown formatting is in the query
        sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
        
        logger.info(f"Executing query: {sql_query}")
            
        # Create connection
        conn = snowflake.connector.connect(
            user=snowflake_config['user'],
            password=snowflake_config['password'],
            account=snowflake_config['account'],
            warehouse=snowflake_config['warehouse'],
            database=snowflake_config['database'],
            schema=snowflake_config['schema']
        )
            
        try:
            # Execute query
            cursor = conn.cursor()
            cursor.execute(sql_query)
                
            # Get results
            if cursor.description:
                columns = [col[0] for col in cursor.description]
                data = cursor.fetchall()
                
                # Create DataFrame with proper type handling
                df = pd.DataFrame(data, columns=columns)
                
                # Replace special float values with None for JSON compatibility
                for col in df.columns:
                    if df[col].dtype.kind == 'f':
                        # Replace NaN and infinity values with None
                        df[col] = df[col].replace([np.nan, np.inf, -np.inf], None)
                    # Convert timestamp columns to ISO format strings
                    elif df[col].dtype.kind == 'M' or 'time' in col.lower() or 'date' in col.lower() or 'ts' in col.lower():
                        df[col] = df[col].apply(lambda x: x.isoformat() if hasattr(x, 'isoformat') else x)
                
                if not is_debug:
                    logger.info(f"Query returned {len(df)} rows")
                return df
            else:
                if not is_debug:
                    logger.info("Query executed successfully (no results)")
                return pd.DataFrame()
                
        finally:
            cursor.close()
            conn.close()
            
    except Exception as e:
        logger.error(f"Error executing query: {str(e)}")
        if not is_debug:
            raise
        return None