import os
from openai import OpenAI
from dotenv import load_dotenv
from schema import get_schema_metadata
from sql_generation import generate_safe_sql
from query_executor import execute_query
from explanation import explain_results

# Load environment variables from .env file
load_dotenv()

class SnowflakeAIAgent:
    def __init__(self):
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.snowflake_config = {
            "user": os.getenv("SNOWFLAKE_USER"),
            "password": os.getenv("SNOWFLAKE_PASSWORD"),
            "account": os.getenv("SNOWFLAKE_ACCOUNT"),
            "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE"),
            "database": os.getenv("SNOWFLAKE_DATABASE"),
            "schema": os.getenv("SNOWFLAKE_SCHEMA"),
            "role": os.getenv("SNOWFLAKE_ROLE", "READ_ONLY_ROLE")
        }
        self.schema_metadata = get_schema_metadata()

    def process_question(self, user_question):
        try:
            sql_query = generate_safe_sql(user_query=user_question, schema_metadata=self.schema_metadata, openai_client=self.openai_client)
            print(f"Generated SQL: {sql_query}")
            results = execute_query(sql_query, self.snowflake_config)
            explanation = explain_results(sql_query, results, user_question, self.openai_client)
            return {
                "success": True,
                "query": sql_query,
                "results": results,
                "explanation": explanation
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            } 