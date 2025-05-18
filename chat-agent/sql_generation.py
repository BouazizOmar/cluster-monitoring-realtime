import json
import re

def clean_sql_query(query):
    """
    Remove comments, leading whitespace, and markdown code blocks from SQL query
    
    Args:
        query: The SQL query to clean
        
    Returns:
        Cleaned SQL query
    """
    if not query:
        return ""
    
    # First, remove any markdown code block indicators (```)
    query = query.replace("```sql", "").replace("```", "").strip()
    
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
    
    # Join cleaned lines with spaces to create a single query string
    return ' '.join([line.strip() for line in cleaned_lines])


def standardize_vm_name(vm_name):
    """
    Standardize VM names to ensure proper matching in queries
    
    Args:
        vm_name: The VM name to standardize
        
    Returns:
        A tuple of (original_format, space_format, underscore_format)
    """
    if not vm_name:
        return None
        
    # Original format (strip whitespace)
    original = vm_name.strip()
    
    # Format with spaces (replace underscores with spaces)
    space_format = original.replace('_', ' ').strip()
    
    # Format with underscores (replace spaces with underscores)
    underscore_format = original.replace(' ', '_').strip()
    
    # Handle common names with correct capitalization
    standard_names = {
        'ubuntu': 'Ubuntu',
        'lubuntu': 'Lubuntu',
        'lubuntu v2': 'Lubuntu V2',
        'lubuntu_v2': 'Lubuntu V2'
    }
    
    # Try to match with standard names (case-insensitive)
    for key, value in standard_names.items():
        if original.lower() == key:
            original = value
        if space_format.lower() == key:
            space_format = value
        if underscore_format.lower() == key:
            underscore_format = value
    
    return (original, space_format, underscore_format)


def generate_safe_sql(user_query, schema_metadata, openai_client):
    """
    Use OpenAI to generate a safe, read-only SQL query based on user's natural language query
    
    Args:
        user_query: Natural language query from user
        schema_metadata: Database schema metadata
        openai_client: OpenAI API client
        
    Returns:
        Tuple containing (main query, debug queries)
    """
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
        
        # Check VM names
        """
        SELECT DISTINCT VM, VM_KEY, OS_TYPE 
        FROM DIM_VM
        ORDER BY VM_KEY;
        """
    ]
    
    # Actual table schema examples from the user's database
    tables_example = """
    FACT_PROCESS_SERVICE_VALUES:
    PROCESS_SERVICE_VALUE_ID | METRIC_KEY | HOST_KEY | MODE_KEY | CPU_KEY | VM_KEY | TIME_KEY | SERVICE_NAME | STATE | TYPE | VALUE | KAFKA_OFFSET | KAFKA_TIMESTAMP | INGEST_TIMESTAMP
    6192 | 70 | 3 |  |  | 2 | 2 | ModemManager.service | active | dbus | 1 | 1108547 | 2025-04-25 13:44:16.000 | 2025-04-25 05:54:12.665
    6193 | 70 | 3 |  |  | 2 | 2 | ModemManager.service | failed | dbus | 0 | 1108548 | 2025-04-25 13:44:16.000 | 2025-04-25 05:54:12.665
    
    FACT_METRIC_VALUES:
    METRIC_VALUE_ID | METRIC_KEY | HOST_KEY | MODE_KEY | CPU_KEY | VM_KEY | TIME_KEY | VALUE | KAFKA_OFFSET | KAFKA_TIMESTAMP | INGEST_TIMESTAMP
    422688 | 1 | 1 | 1 | 4 | 1 | 3607 | 0 | 1296760 | 2025-04-28 13:40:40.000 | 2025-04-28 05:43:38.136
    422689 | 1 | 1 | 2 | 4 | 1 | 3607 | 0 | 1296761 | 2025-04-28 13:40:40.000 | 2025-04-28 05:43:38.136
    
    DIM_CPU:
    CPU_KEY | CPU_ID
    3 | 3
    1 | 1
    4 | 0
    2 | 2
    
    DIM_HOST:
    HOST_KEY | INSTANCE | JOB
    1 | 10.71.0.59:9102 | node
    2 | 10.71.0.59:9103 | node
    3 | 10.71.0.59:9101 | node
    
    DIM_MODE:
    MODE_KEY | MODE
    3 | iowait
    7 | system
    8 | softirq
    4 | steal
    5 | idle
    
    DIM_TIME:
    TIME_KEY | TS | DAY | HOUR | MINUTE
    17696 | 2025-05-06 15:46:41.000 | 2025-05-06 | 15 | 46
    17691 | 2025-05-06 15:46:56.000 | 2025-05-06 | 15 | 46
    
    DIM_VM:
    VM_KEY | VM | OS_TYPE
    1 | Ubuntu | linux
    2 | Lubuntu | linux
    3 | Lubuntu V2 | linux
    
    DIM_METRIC (partial - important metrics only):
    METRIC_KEY | METRIC_NAME
    26 | node_memory_Active_bytes
    41 | node_memory_MemTotal_bytes
    65 | node_memory_Cached_bytes
    66 | node_memory_MemFree_bytes
    78 | node_memory_MemAvailable_bytes
    """
    
    prompt = f"""
    You are a SQL query generator for a Snowflake database with a galaxy schema focused on system metrics monitoring.

    Here's the detailed table schema with actual example data from the database:
    {tables_example}

    Additional schema metadata:
    {schema_context}

    This is a monitoring system for metrics from various hosts, CPUs, VMs, and services. The schema follows a snowflake design:

    FACT TABLES:
    - FACT_METRIC_VALUES: Contains numeric metric measurements (VALUE field) with METRIC_KEY, HOST_KEY, CPU_KEY, VM_KEY, TIME_KEY
    - FACT_PROCESS_SERVICE_VALUES: Contains service metrics and states

    DIMENSION TABLES:
    - DIM_METRIC: Contains metric names and definitions
    - DIM_HOST: Contains host information (instance, job)
    - DIM_CPU: Contains CPU identifiers (CPU_KEY, CPU_ID)
    - DIM_VM: Contains VM identifiers and OS types (VM_KEY, VM, OS_TYPE)
    - DIM_MODE: Contains operation modes
    - DIM_TIME: Contains time hierarchy (timestamp, day, hour, minute)

    EXAMPLES OF CORRECT QUERIES:

    1. Query: "Show memory usage for each VM"
    SQL:
    ```
    SELECT 
      dvm.VM AS "VM",
      AVG(CASE 
        WHEN mt_fmv.VALUE > 0 THEN ((mt_fmv.VALUE - ma_fmv.VALUE) / NULLIF(mt_fmv.VALUE, 0)) * 100 
        ELSE NULL 
      END) AS "AVG_MEMORY_USAGE_PCT"
    FROM 
      DIM_VM dvm
    JOIN 
      FACT_METRIC_VALUES mt_fmv ON dvm.VM_KEY = mt_fmv.VM_KEY
    JOIN 
      DIM_METRIC mt ON mt_fmv.METRIC_KEY = mt.METRIC_KEY 
    JOIN 
      FACT_METRIC_VALUES ma_fmv ON dvm.VM_KEY = ma_fmv.VM_KEY AND mt_fmv.TIME_KEY = ma_fmv.TIME_KEY
    JOIN 
      DIM_METRIC ma ON ma_fmv.METRIC_KEY = ma.METRIC_KEY 
    WHERE
      mt.METRIC_NAME = 'node_memory_MemTotal_bytes'
      AND ma.METRIC_NAME = 'node_memory_MemAvailable_bytes'
      AND mt_fmv.VALUE IS NOT NULL AND mt_fmv.VALUE::STRING != 'NaN'
      AND ma_fmv.VALUE IS NOT NULL AND ma_fmv.VALUE::STRING != 'NaN'
    GROUP BY 
      dvm.VM
    ORDER BY
      dvm.VM;
    ```
    
    3. Query: "Give me the top failed service in VM Ubuntu"
    SQL:
    ```
    SELECT 
        fpsv.SERVICE_NAME,
        COUNT(*) AS FAILURES
    FROM 
        FACT_PROCESS_SERVICE_VALUES fpsv
    JOIN 
        DIM_VM dvm ON fpsv.VM_KEY = dvm.VM_KEY
    WHERE 
        dvm.VM = 'Ubuntu' AND fpsv.STATE = 'failed'
    GROUP BY 
        fpsv.SERVICE_NAME
    ORDER BY 
        FAILURES DESC
    LIMIT 1;
    ```

    2. Query: "Show me average CPU usage by mode for Lubuntu"
    SQL:
    ```
    WITH cpu_samples AS (
      SELECT
        f.MODE_KEY,
        dm.MODE,
        f.VALUE,
        dt.TS
      FROM FACT_METRIC_VALUES f
      JOIN DIM_METRIC m ON f.METRIC_KEY = m.METRIC_KEY
      JOIN DIM_VM v ON f.VM_KEY = v.VM_KEY
      JOIN DIM_MODE dm ON f.MODE_KEY = dm.MODE_KEY
      JOIN DIM_TIME dt ON f.TIME_KEY = dt.TIME_KEY
      WHERE v.VM = 'Lubuntu'
        AND m.METRIC_NAME = 'node_cpu_seconds_total'
        AND f.VALUE IS NOT NULL
        AND f.VALUE::STRING != 'NaN'
    )
    , mode_stats AS (
      SELECT
        MODE_KEY,
        MODE,
        COUNT(*) AS sample_count,
        MIN(VALUE) AS min_val,
        MAX(VALUE) AS max_val,
        DATEDIFF('second', MIN(TS), MAX(TS)) AS time_diff_seconds
      FROM cpu_samples
      GROUP BY MODE_KEY, MODE
    )
    SELECT
      MODE,
      sample_count,
      min_val,
      max_val,
      time_diff_seconds,
      CASE
        WHEN time_diff_seconds > 0 THEN ROUND(100 * (max_val - min_val) / NULLIF(time_diff_seconds, 0), 2)
        ELSE NULL
      END AS avg_cpu_pct
    FROM mode_stats
    ORDER BY avg_cpu_pct DESC;
    ```
    
    4. Query: "Show me all VM instances"
    SQL:
    ```
    SELECT 
        VM_KEY,
        VM AS "VM_NAME", 
        OS_TYPE
    FROM 
        DIM_VM 
    ORDER BY 
        VM_KEY;
    ```

    The user's query is: "{user_query}"

    IMPORTANT RULES:
    1. Generate ONLY a SQL query (must start with SELECT or WITH).
    2. Use EXACT table and column names from the schema.
    3. Include proper JOIN conditions between fact and dimension tables.
    4. When filtering by VM name, use EXACT string matching (WHERE dvm.VM = 'Ubuntu').
    5. Only use fuzzy matching with LIKE for intentional partial matches.
    6. For memory usage calculations, ALWAYS use both 'node_memory_MemTotal_bytes' AND 'node_memory_MemAvailable_bytes'.
    7. Handle NULL values properly using CASE statements and NULLIF to prevent division by zero.
    8. ALWAYS filter out NaN values: Snowflake has no IS_NAN() function, so use: WHERE VALUE IS NOT NULL AND VALUE::STRING != 'NaN'
    9. Return only the SQL query with no comments, explanations, or formatting.
    10. Include JOIN to DIM_TIME for time-based queries.
    11. For CPU metrics, check if the specific CPU_KEY is needed in the JOIN conditions.

    Based on the user's query, generate the most appropriate Snowflake SQL query.
    """
    
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a SQL expert that generates precise, optimized queries based on galaxy schema databases."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1
    )
    
    sql_query = response.choices[0].message.content.strip()
    sql_query_clean = clean_sql_query(sql_query)
    
    # Only check for dangerous modifications (no default query fallback)
    dangerous_keywords = ["INSERT", "UPDATE", "DELETE", "CREATE", "ALTER", "DROP", "TRUNCATE", "MERGE", "EXECUTE", "GRANT", "REVOKE"]
    if any(f" {keyword} " in f" {sql_query_clean.upper()} " for keyword in dangerous_keywords):
        raise ValueError(f"Generated query contains disallowed data modification keywords: {sql_query}")
        
    # Log for debugging
    print(f"Generated query: {sql_query_clean[:100]}...")
    
    return sql_query_clean, debug_queries