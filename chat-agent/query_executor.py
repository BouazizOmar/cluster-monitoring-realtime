import pandas as pd
from snowflake.connector import connect as snowflake_connect
from sqlalchemy import create_engine
import logging
import snowflake.connector

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def snowflake_connection(snowflake_config):
    """Create a Snowflake connection"""
    try:
        return snowflake_connect(**snowflake_config)
    except Exception as e:
        logger.error(f"Failed to connect to Snowflake: {str(e)}")
        raise

def create_sqlalchemy_engine(snowflake_config):
    """Create a SQLAlchemy engine for Snowflake"""
    try:
        connection_string = f"snowflake://{snowflake_config['user']}:{snowflake_config['password']}@{snowflake_config['account']}/{snowflake_config['database']}/{snowflake_config['schema']}?warehouse={snowflake_config['warehouse']}&role={snowflake_config['role']}"
        return create_engine(connection_string)
    except Exception as e:
        logger.error(f"Failed to create SQLAlchemy engine: {str(e)}")
        raise

def execute_query(sql_query, snowflake_config, is_debug=False):
    """Execute a SQL query and return results as a pandas DataFrame"""
    try:
        # Extract the query if it's in a tuple
        if isinstance(sql_query, tuple):
            sql_query = sql_query[0]
            
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
                df = pd.DataFrame(data, columns=columns)
                
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

def execute_debug_queries(debug_queries, snowflake_config):
    """Execute debug queries and return their results"""
    results = []
    for i, query in enumerate(debug_queries, 1):
        try:
            # Extract the query if it's in a tuple
            if isinstance(query, tuple):
                query = query[0]
            
            logger.info(f"\nExecuting Debug Query {i}:")
            logger.info(f"Query: {query}")
            
            # Execute the query
            result = execute_query(query, snowflake_config, is_debug=True)
            
            if result is not None and not result.empty:
                logger.info(f"Debug Query {i} Results:")
                logger.info(result.to_string())
                results.append(result)
            else:
                logger.warning(f"Debug Query {i} returned no results")
                results.append(None)
                
        except Exception as e:
            logger.error(f"Error executing debug query {i}: {str(e)}")
            results.append(None)
            
    return results 