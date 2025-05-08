def explain_results(query, results_df, user_question, openai_client):
    """Generate an explanation of the query results using OpenAI"""
    if results_df is not None and not results_df.empty:
        sample_df = results_df.head(10) if len(results_df) > 10 else results_df
        result_str = sample_df.to_string()
        row_count = len(results_df)
    else:
        result_str = "No results or empty dataset"
        row_count = 0
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

    Please provide:
    1. A clear explanation of these results in relation to the user's question
    2. Any important insights from the data
    3. Potential follow-up questions the user might want to ask
    '''
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a data analyst explaining query results clearly and helpfully."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )
    return response.choices[0].message.content 