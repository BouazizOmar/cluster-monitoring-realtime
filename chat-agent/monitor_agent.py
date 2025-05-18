import os
from openai import OpenAI
from dotenv import load_dotenv
import snowflake.connector
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Optional, Dict, Any
from schema_description import SCHEMA_DESCRIPTION
import pandas as pd

# Load environment variables
load_dotenv()

# Define the state for the LangGraph workflow
class AgentState(TypedDict):
    user_input: str
    conversation_history: List[dict]
    sql_queries: Optional[List[str]]
    columns: Optional[List[List[str]]]
    results: Optional[List[List]]  # Raw rows from DB
    dataframes: Optional[List[pd.DataFrame]]  # Pandas dataframes
    diagnostic_info: Optional[Dict[str, Any]]  # Diagnostic information
    output: Optional[str]  # Final markdown output

# Snowflake connection parameters
SNOWFLAKE_CONFIG = {
    "user": os.getenv("SNOWFLAKE_USER"),
    "password": os.getenv("SNOWFLAKE_PASSWORD"),
    "account": os.getenv("SNOWFLAKE_ACCOUNT"),
    "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE"),
    "database": os.getenv("SNOWFLAKE_DATABASE"),
    "schema": os.getenv("SNOWFLAKE_SCHEMA"),
    "role": os.getenv("SNOWFLAKE_ROLE", "READ_ONLY_ROLE"),
}

# Configure OpenAI API
# Simple initialization compatible with OpenAI SDK v1.3.x
try:
    # Try to instantiate with only the API key for simplicity
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except TypeError as e:
    # If that fails, try an alternative initialization
    import httpx
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        http_client=httpx.Client(timeout=60.0)
    )

def clean_sql_query(query: str) -> str:
    """
    Remove comments and leading whitespace from SQL query
    """
    if not query:
        return ""
        
    lines = query.split('\n')
    cleaned_lines = []
    in_block_comment = False
    for line in lines:
        if in_block_comment:
            if '*/' in line:
                line = line[line.find('*/') + 2:]
                in_block_comment = False
            else:
                continue
        if '/*' in line:
            if '*/' in line[line.find('/*'):]:
                before = line[:line.find('/*')]
                after = line[line.find('*/') + 2:]
                line = before + after
            else:
                line = line[:line.find('/*')]
                in_block_comment = True
        if '--' in line:
            line = line[:line.find('--')]
        if line.strip():
            cleaned_lines.append(line)
    
    return ' '.join([line.strip() for line in cleaned_lines])

def generate_sql_query(state: AgentState) -> AgentState:
    """Generate SQL queries from user input using OpenAI."""
    user_input = state["user_input"]
    conversation_history = state["conversation_history"]
    
    # Create conversation history text for context
    history_text = ""
    if conversation_history:
        history_text = "\n".join([
            f"User: {msg['user']}\nSQL Query: {msg.get('sql', 'None')}\nResponse: {msg.get('output', 'No response')[:200]}..."
            for msg in conversation_history[-3:]  # Only include the last 3 exchanges for brevity
        ])
    
    # Prepare prompt for SQL generation
    prompt = f"""
You are an expert SQL query generator for a Snowflake database with a galaxy schema focused on system metrics monitoring.
Based on the following schema and conversation history, convert the user's natural language query into a valid SQL query.

SCHEMA DESCRIPTION:
{SCHEMA_DESCRIPTION}

CONVERSATION HISTORY:
{history_text}

USER QUERY: {user_input}

IMPORTANT RULES:
1. Generate ONLY a SQL query (must start with SELECT).
2. Use EXACT table and column names from the schema.
3. Include proper JOIN conditions between fact and dimension tables.
4. When filtering by VM name, use EXACT string matching (WHERE dvm.VM = 'Ubuntu').
5. Only use fuzzy matching with LIKE for intentional partial matches.
6. For memory usage calculations, ALWAYS use both 'node_memory_MemTotal_bytes' AND 'node_memory_MemAvailable_bytes'.
7. Handle NULL values properly using CASE statements and NULLIF to prevent division by zero.
8. Return only the SQL query with no comments, explanations, or formatting.
9. If joining multiple fact tables, ensure proper time alignment.

SQL QUERY:
"""
    
    # Call OpenAI API
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a SQL expert that generates precise, optimized queries based on galaxy schema databases."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1
    )
    
    # Extract and clean the SQL query
    sql_query = response.choices[0].message.content.strip()
    if sql_query.startswith('```sql') and sql_query.endswith('```'):
        sql_query = sql_query[6:-3].strip()
    elif sql_query.startswith('```') and sql_query.endswith('```'):
        sql_query = sql_query[3:-3].strip()
    
    sql_query_clean = clean_sql_query(sql_query)
    
    # Basic validation
    if not sql_query_clean.upper().startswith("SELECT"):
        default_query = """
        SELECT VM AS "VM", VM_KEY, OS_TYPE
        FROM DIM_VM
        ORDER BY VM;
        """
        print(f"Warning: Generated query doesn't start with SELECT. Using default query instead.")
        print(f"Original query was: {sql_query}")
        sql_query_clean = default_query
    
    # Check for dangerous SQL
    dangerous_keywords = ["INSERT", "UPDATE", "DELETE", "CREATE", "ALTER", "DROP", "TRUNCATE", "MERGE", "EXECUTE", "GRANT", "REVOKE"]
    if any(f" {keyword} " in f" {sql_query_clean.upper()} " for keyword in dangerous_keywords):
        print(f"Warning: Query contains dangerous keywords. Using safe default query instead.")
        sql_query_clean = "SELECT 'Access Denied' AS message FROM DUAL;"
    
    state["sql_queries"] = [sql_query_clean]
    return state

def execute_snowflake_query(state: AgentState) -> AgentState:
    """Execute SQL queries on Snowflake and return results."""
    sql_queries = state["sql_queries"]
    all_results = []
    all_columns = []
    diagnostic_info = {}

    try:
        conn = snowflake.connector.connect(**SNOWFLAKE_CONFIG)
        cursor = conn.cursor()
        
        # Execute each query
        for sql_query in sql_queries:
            try:
                cursor.execute(sql_query)
                results = cursor.fetchall()
                columns = [desc[0] for desc in cursor.description]
                all_results.append(results)
                all_columns.append(columns)
            except Exception as e:
                print(f"Error executing query: {e}")
                all_results.append(f"Error: {str(e)}")
                all_columns.append([])
        
        # If the queries returned no results, get diagnostic information
        if not all_results or (isinstance(all_results[0], list) and len(all_results[0]) == 0):
            # Get available VM names for reference
            cursor.execute("SELECT DISTINCT VM, VM_KEY FROM DIM_VM ORDER BY VM")
            vm_df = cursor.fetchall()
            vm_columns = [desc[0] for desc in cursor.description]
            
            # Get available metrics
            if 'memory' in state["user_input"].lower():
                metrics_filter = "METRIC_NAME LIKE '%memory%' OR METRIC_NAME LIKE '%mem%'"
            elif 'cpu' in state["user_input"].lower():
                metrics_filter = "METRIC_NAME LIKE '%cpu%'"
            else:
                metrics_filter = "1=1 LIMIT 20"
                
            cursor.execute(f"SELECT DISTINCT METRIC_KEY, METRIC_NAME FROM DIM_METRIC WHERE {metrics_filter}")
            metric_df = cursor.fetchall()
            metric_columns = [desc[0] for desc in cursor.description]
            
            diagnostic_info = {
                "vm_names": [row[0] for row in vm_df],
                "metrics": [row[1] for row in metric_df],
                "vm_data": {"rows": vm_df, "columns": vm_columns},
                "metric_data": {"rows": metric_df, "columns": metric_columns},
            }
        
        conn.close()
        
    except Exception as e:
        print(f"Database connection error: {e}")
        all_results = [f"Database connection error: {str(e)}"]
        all_columns = [[]]
    
    state["columns"] = all_columns
    state["results"] = all_results
    state["diagnostic_info"] = diagnostic_info
    return state

def convert_to_dataframe(state: AgentState) -> AgentState:
    """Convert query results into pandas DataFrames."""
    columns = state["columns"]
    results = state["results"]
    sql_queries = state["sql_queries"]
    dataframes = []

    for i, (query, cols, res) in enumerate(zip(sql_queries, columns, results)):
        if isinstance(res, str) or not res or not cols:
            dataframes.append(pd.DataFrame())
        else:
            df = pd.DataFrame(res, columns=cols)
            dataframes.append(df)
    
    # Also convert diagnostic info to dataframes if available
    if state["diagnostic_info"]:
        diag_info = state["diagnostic_info"]
        if "vm_data" in diag_info and diag_info["vm_data"]["rows"]:
            diag_info["vm_dataframe"] = pd.DataFrame(
                diag_info["vm_data"]["rows"], 
                columns=diag_info["vm_data"]["columns"]
            )
        if "metric_data" in diag_info and diag_info["metric_data"]["rows"]:
            diag_info["metric_dataframe"] = pd.DataFrame(
                diag_info["metric_data"]["rows"], 
                columns=diag_info["metric_data"]["columns"]
            )
        state["diagnostic_info"] = diag_info

    state["dataframes"] = dataframes
    return state

def generate_explanation(state: AgentState) -> AgentState:
    """Generate a natural language explanation of the query results."""
    sql_queries = state["sql_queries"]
    dataframes = state["dataframes"]
    user_input = state["user_input"]
    diagnostic_info = state["diagnostic_info"]
    
    # Prepare dataframe information for the prompt
    df_samples = []
    for i, (query, df) in enumerate(zip(sql_queries, dataframes)):
        if not df.empty:
            df_sample = df.head(5).to_string()
            row_count = len(df)
            df_samples.append(f"Query {i+1}: {query}\nRows: {row_count}\nSample data:\n{df_sample}")
        else:
            df_samples.append(f"Query {i+1}: {query}\nNo results returned.")
    
    df_info = "\n\n".join(df_samples)
    
    # Add diagnostic information if available
    diag_info = ""
    if diagnostic_info:
        vm_names = ", ".join(diagnostic_info.get("vm_names", []))
        metrics = ", ".join(diagnostic_info.get("metrics", []))
        diag_info = f"\nDiagnostic Information:\n- Available VM names: {vm_names}\n- Available metrics: {metrics}"
    
    # Prepare the prompt for explanation generation
    prompt = f"""
You are an expert at interpreting database query results and explaining them in plain language.
Given the user's query, SQL query, and results below, provide a clear explanation of what the data shows.

User Query: {user_input}

SQL Query and Results:
{df_info}
{diag_info}

Instructions:
1. Explain what the data shows in relation to the user's question
2. If no results were returned, explain possible reasons why
3. Provide important insights or patterns in the data if they exist
4. Focus on the most important aspects of the data, especially memory usage patterns if applicable
5. Be concise but informative - aim for 2-3 paragraphs maximum

Explanation:
"""
    
    # Call OpenAI API for explanation
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a data analyst explaining query results clearly and helpfully."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )
    
    explanation = response.choices[0].message.content.strip()
    
    # Format the output with both query results and explanation
    output_parts = []
    
    # Add query and results
    for i, (query, df) in enumerate(zip(sql_queries, dataframes)):
        output_parts.append(f"SQL Query {i+1}:\n{query}\n")
        if not df.empty:
            output_parts.append(f"Results ({len(df)} rows):\n{df.head(10).to_string()}")
            if len(df) > 10:
                output_parts.append(f"Showing 10 of {len(df)} rows\n")
        else:
            output_parts.append("No results returned\n")
    
    # Add explanation
    output_parts.append(f"Explanation:\n{explanation}")
    
    # Set output in state
    state["output"] = "\n\n".join(output_parts)
    
    # Update conversation history
    state["conversation_history"].append({
        "user": state["user_input"],
        "sql": "; ".join(sql_queries),
        "output": explanation
    })
    
    return state

def build_graph():
    """Build the LangGraph workflow."""
    workflow = StateGraph(AgentState)

    # Define nodes
    workflow.add_node("generate_sql", generate_sql_query)
    workflow.add_node("execute_query", execute_snowflake_query)
    workflow.add_node("convert_to_dataframe", convert_to_dataframe)
    workflow.add_node("generate_explanation", generate_explanation)

    # Define edges
    workflow.add_edge("generate_sql", "execute_query")
    workflow.add_edge("execute_query", "convert_to_dataframe")
    workflow.add_edge("convert_to_dataframe", "generate_explanation")
    workflow.add_edge("generate_explanation", END)

    # Set entry point
    workflow.set_entry_point("generate_sql")

    return workflow.compile()

def process_query(user_input: str, conversation_history: List[dict] = None) -> dict:
    """Process a single user query and return the results."""
    if not user_input:
        return {"sql_queries": [], "output": "Please enter a valid query."}

    if conversation_history is None:
        conversation_history = []

    # Initialize the graph
    graph = build_graph()
    
    # Create initial state
    state = AgentState(
        user_input=user_input,
        conversation_history=conversation_history,
        sql_queries=None,
        columns=None,
        results=None,
        dataframes=None,
        diagnostic_info=None,
        output=None
    )

    # Run the graph
    final_state = graph.invoke(state)
    
    return {
        "sql_queries": final_state["sql_queries"],
        "output": final_state["output"],
        "dataframes": final_state["dataframes"],
        "conversation_history": final_state["conversation_history"]
    }

def main():
    """Main function to handle user interaction and query processing."""
    print("Welcome to the Monitoring System Query Agent! Type 'exit' to quit.")
    conversation_history = []

    while True:
        user_input = input("\nEnter your query: ").strip()
        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        try:
            result = process_query(user_input, conversation_history)
            print("\n" + result["output"])
            conversation_history = result["conversation_history"]
        except Exception as e:
            print(f"Error processing query: {e}")

if __name__ == "__main__":
    main() 