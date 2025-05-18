#!/usr/bin/env python3
"""
Example client for the monitoring pipeline API
"""
import requests
import json
from typing import Dict, Any, Optional

API_URL = "http://localhost:9000"  # Updated to use port 9000

def call_original_api(query: str) -> Dict[str, Any]:
    """
    Call the original SnowflakeAIAgent API endpoint
    """
    url = f"{API_URL}/api/process-query"
    payload = {"query": query}
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()  # Raise exception for 4XX/5XX responses
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error calling original API: {e}")
        if hasattr(e, 'response') and e.response:
            print(f"Response text: {e.response.text}")
        return {"success": False, "error": str(e)}

def call_langgraph_api(query: str, conversation_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Call the new LangGraph implementation API endpoint
    """
    url = f"{API_URL}/api/langgraph-query"
    payload = {"query": query}
    
    if conversation_id:
        payload["conversation_id"] = conversation_id
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()  # Raise exception for 4XX/5XX responses
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error calling LangGraph API: {e}")
        if hasattr(e, 'response') and e.response:
            print(f"Response text: {e.response.text}")
        return {"success": False, "error": str(e)}

def list_conversations() -> Dict[str, Any]:
    """
    List all active conversations
    """
    url = f"{API_URL}/api/conversations"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error listing conversations: {e}")
        return {"success": False, "error": str(e)}

def delete_conversation(conversation_id: str) -> Dict[str, Any]:
    """
    Delete a conversation by ID
    """
    url = f"{API_URL}/api/conversations/{conversation_id}"
    
    try:
        response = requests.delete(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error deleting conversation: {e}")
        return {"success": False, "error": str(e)}

def print_result(result: Dict[str, Any], api_type: str):
    """
    Pretty print the API result
    """
    print(f"\n==== {api_type} API Result ====")
    if not result.get("success", False):
        print(f"ERROR: {result.get('error', 'Unknown error')}")
        return
    
    if api_type == "Original":
        print(f"SQL Query: {result.get('sql_query', 'No query available')}")
        print("\nExplanation:")
        print(result.get("explanation", "No explanation available"))
        
        # Print sample results if available
        if "results" in result and "rows" in result["results"] and result["results"]["rows"]:
            print("\nSample Results:")
            rows = result["results"]["rows"][:5]  # Show up to 5 rows
            for row in rows:
                print(json.dumps(row, indent=2))
            print(f"... {result['results'].get('row_count', len(rows))} total rows")
    else:  # LangGraph API
        print(f"Conversation ID: {result.get('conversation_id', 'No ID available')}")
        print(f"SQL Queries: {result.get('sql_queries', ['No queries available'])}")
        print("\nOutput:")
        print(result.get("output", "No output available"))
        
        # Print dataframe results if available
        if "dataframe_results" in result and result["dataframe_results"]:
            print("\nDataframe Results:")
            for df_result in result["dataframe_results"]:
                print(f"\nDataframe {df_result['index']}:")
                rows = df_result["rows"][:3]  # Show up to 3 rows
                for row in rows:
                    print(json.dumps(row, indent=2))
                print(f"... {df_result.get('row_count', len(rows))} total rows")

def demo_original_api():
    """
    Demonstrate the original API
    """
    print("\n\n========== ORIGINAL API DEMO ==========")
    query = "Show me the memory usage for Ubuntu VM"
    print(f"Query: {query}")
    
    result = call_original_api(query)
    print_result(result, "Original")

def demo_langgraph_api_with_conversation():
    """
    Demonstrate the LangGraph API with conversation history
    """
    print("\n\n========== LANGGRAPH API WITH CONVERSATION DEMO ==========")
    
    # First query
    query1 = "Show me the memory usage for Ubuntu VM"
    print(f"Query 1: {query1}")
    result1 = call_langgraph_api(query1)
    print_result(result1, "LangGraph")
    
    # Get the conversation ID from the first result
    conversation_id = result1.get("conversation_id")
    if not conversation_id:
        print("Error: No conversation ID returned. Cannot continue with conversation.")
        return
    
    # Second query referencing the first
    query2 = "Compare that with Lubuntu"
    print(f"\nQuery 2 (using same conversation): {query2}")
    result2 = call_langgraph_api(query2, conversation_id)
    print_result(result2, "LangGraph")
    
    # Third query referencing both previous queries
    query3 = "Which one has better memory usage?"
    print(f"\nQuery 3 (using same conversation): {query3}")
    result3 = call_langgraph_api(query3, conversation_id)
    print_result(result3, "LangGraph")
    
    # List all conversations
    print("\nListing all conversations:")
    conversations = list_conversations()
    print(json.dumps(conversations, indent=2))
    
    # Delete the conversation
    print(f"\nDeleting conversation {conversation_id}:")
    delete_result = delete_conversation(conversation_id)
    print(json.dumps(delete_result, indent=2))

def main():
    """
    Main function to demonstrate API calls
    """
    # Demonstrate the original API
    demo_original_api()
    
    # Demonstrate the LangGraph API with conversation history
    demo_langgraph_api_with_conversation()

if __name__ == "__main__":
    main() 