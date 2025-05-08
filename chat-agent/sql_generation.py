def clean_sql_query(query):
    """Remove comments and leading whitespace from SQL query"""
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


def generate_safe_sql(user_query, schema_metadata, openai_client):
    """Use OpenAI to generate a safe, read-only SQL query based on user's natural language query"""
    import json
    import re
    schema_context = json.dumps(schema_metadata, indent=2)
    
    # Debug queries to check data and relationships
    debug_queries = [
        # Check table counts
        """
        SELECT 
            'FACT_METRIC_VALUES' as table_name, COUNT(*) as count 
        FROM FACT_METRIC_VALUES
        UNION ALL
        SELECT 
            'DIM_METRIC' as table_name, COUNT(*) as count 
        FROM DIM_METRIC
        UNION ALL
        SELECT 
            'DIM_CPU' as table_name, COUNT(*) as count 
        FROM DIM_CPU
        UNION ALL
        SELECT 
            'DIM_TIME' as table_name, COUNT(*) as count 
        FROM DIM_TIME;
        """,
        
        # Check CPU metrics
        """
        SELECT DISTINCT METRIC_NAME 
        FROM DIM_METRIC 
        WHERE METRIC_NAME LIKE '%cpu%';
        """,
        
        # Check sample data with joins
        """
        SELECT 
            fmv.METRIC_KEY,
            dm.METRIC_NAME,
            fmv.CPU_KEY,
            dc.CPU_ID,
            fmv.VALUE,
            fmv.TIME_KEY
        FROM 
            FACT_METRIC_VALUES fmv
        JOIN 
            DIM_METRIC dm ON fmv.METRIC_KEY = dm.METRIC_KEY
        JOIN 
            DIM_CPU dc ON fmv.CPU_KEY = dc.CPU_KEY
        LIMIT 5;
        """
    ]
    
    # Extract time period from query
    time_period = None
    time_patterns = {
        r'last (\d+) days?': ('day', 'DAY'),
        r'last (\d+) weeks?': ('week', 'DAY'),
        r'last (\d+) months?': ('month', 'DAY'),
        r'last (\d+) hours?': ('hour', 'HOUR')
    }
    
    for pattern, (unit, time_col) in time_patterns.items():
        match = re.search(pattern, user_query.lower())
        if match:
            value = int(match.group(1))
            if unit == 'week':
                value *= 7  # Convert weeks to days
            time_period = (value, unit, time_col)
            break
    
    prompt = f"""
    You are a SQL query generator for a Snowflake database with a galaxy schema focused on system and service metrics monitoring.

    Here's the metadata about the schema structure:
    {schema_context}

    This is a monitoring system for metrics from various hosts, CPUs, VMs, and services. The schema has:

    FACT TABLES:
    - FACT_METRIC_VALUES: Contains numeric metric measurements (VALUE field) with METRIC_KEY, HOST_KEY, CPU_KEY, VM_KEY, TIME_KEY
    - FACT_PROCESS_SERVICE_VALUES: Contains service metrics and states

    DIMENSION TABLES:
    - DIM_METRIC: Contains metric names and definitions
    - DIM_HOST: Contains host information (instance, job)
    - DIM_CPU: Contains CPU identifiers (CPU_KEY, CPU_ID)
    - DIM_VM: Contains VM identifiers and OS types
    - DIM_MODE: Contains operation modes
    - DIM_TIME: Contains time hierarchy (timestamp, day, hour, minute)

    The user's query is: "{user_query}"

    IMPORTANT:
    - Generate ONLY a read-only SQL query (must start with SELECT).
    - The query must work with the schema provided.
    - Include the exact table and column names as provided.
    - Include proper JOIN conditions between fact and dimension tables.
    - For time-based queries, use the DIM_TIME table and its columns (DAY, HOUR, MINUTE) appropriately.
    - When filtering by time periods, use DATEADD() function with the correct interval.
    - NO comments, explanations, or markdown formatting - ONLY the SQL query.
    - NO DDL or DML statements (INSERT, UPDATE, DELETE, CREATE, ALTER, DROP, TRUNCATE, MERGE).
    - For CPU usage queries, make sure to join with DIM_METRIC to filter for CPU-related metrics.
    - CPU metrics include: node_cpu_seconds_total, node_cpu_guest_seconds_total, node_cpu_online, node_pressure_cpu_waiting_seconds_total
    - Service names are stored in FACT_PROCESS_SERVICE_VALUES.SERVICE_NAME
    - TIME_KEY is a numeric value that maps to DIM_TIME table

    Example CPU usage query:
    SELECT 
        fpsv.SERVICE_NAME,
        dm.METRIC_NAME,
        AVG(fmv.VALUE) as avg_cpu_usage
    FROM 
        FACT_METRIC_VALUES fmv
    JOIN 
        DIM_METRIC dm ON fmv.METRIC_KEY = dm.METRIC_KEY
    JOIN 
        DIM_CPU dc ON fmv.CPU_KEY = dc.CPU_KEY
    JOIN 
        DIM_TIME dt ON fmv.TIME_KEY = dt.TIME_KEY
    JOIN 
        FACT_PROCESS_SERVICE_VALUES fpsv ON fmv.HOST_KEY = fpsv.HOST_KEY 
        AND fmv.TIME_KEY = fpsv.TIME_KEY
    WHERE 
        dm.METRIC_NAME IN ('node_cpu_seconds_total', 'node_cpu_guest_seconds_total', 'node_cpu_online', 'node_pressure_cpu_waiting_seconds_total')
        AND dt.DAY >= DATEADD(day, -7, CURRENT_DATE())
    GROUP BY 
        fpsv.SERVICE_NAME, dm.METRIC_NAME
    ORDER BY 
        avg_cpu_usage DESC
    LIMIT 1;
    """
    
    response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a SQL expert that only generates valid, read-only SELECT queries with no comments or explanations."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1
    )
    
    sql_query = response.choices[0].message.content.strip()
    sql_query_clean = clean_sql_query(sql_query)
    
    # Validate time period in generated query
    if time_period:
        value, unit, time_col = time_period
        expected_dateadd = f"DATEADD({unit}, -{value}"
        if expected_dateadd not in sql_query_clean:
            print(f"Warning: Query might not properly handle {value} {unit} time period")
            # Add time filter if missing
            if "WHERE" in sql_query_clean:
                sql_query_clean = sql_query_clean.replace("WHERE", f"WHERE dt.{time_col} >= DATEADD({unit}, -{value}, CURRENT_DATE()) AND")
            else:
                sql_query_clean = sql_query_clean.replace("FROM", f"FROM\nWHERE dt.{time_col} >= DATEADD({unit}, -{value}, CURRENT_DATE())")
    
    if not sql_query_clean.upper().startswith("SELECT"):
        default_query = """
        SELECT 
            fpsv.SERVICE_NAME,
            dm.METRIC_NAME,
            AVG(fmv.VALUE) as avg_cpu_usage
        FROM 
            FACT_METRIC_VALUES fmv
        JOIN 
            DIM_METRIC dm ON fmv.METRIC_KEY = dm.METRIC_KEY
        JOIN 
            DIM_CPU dc ON fmv.CPU_KEY = dc.CPU_KEY
        JOIN 
            DIM_TIME dt ON fmv.TIME_KEY = dt.TIME_KEY
        JOIN 
            FACT_PROCESS_SERVICE_VALUES fpsv ON fmv.HOST_KEY = fpsv.HOST_KEY 
            AND fmv.TIME_KEY = fpsv.TIME_KEY
        WHERE 
            dm.METRIC_NAME IN ('node_cpu_seconds_total', 'node_cpu_guest_seconds_total', 'node_cpu_online', 'node_pressure_cpu_waiting_seconds_total')
            AND dt.DAY >= DATEADD(day, -7, CURRENT_DATE())
        GROUP BY 
            fpsv.SERVICE_NAME, dm.METRIC_NAME
        ORDER BY 
            avg_cpu_usage DESC
        LIMIT 1;
        """
        print(f"Warning: Generated query doesn't start with SELECT. Using default query instead.")
        print(f"Original query was: {sql_query}")
        return default_query, debug_queries
    
    dangerous_keywords = ["INSERT", "UPDATE", "DELETE", "CREATE", "ALTER", "DROP", "TRUNCATE", "MERGE", "EXECUTE", "GRANT", "REVOKE"]
    if any(f" {keyword} " in f" {sql_query_clean.upper()} " for keyword in dangerous_keywords):
        raise ValueError(f"Generated query contains disallowed data modification keywords: {sql_query}")
    
    return sql_query_clean, debug_queries 