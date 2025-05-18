from agent import SnowflakeAIAgent
import sys

if __name__ == "__main__":
    agent = SnowflakeAIAgent()
    
    # Check if a question was provided as a command line argument
    if len(sys.argv) > 1:
        question = sys.argv[1]
    else:
        question = "The most used service in Lubuntu VM"
        
    result = agent.process_question(question)
    if result["success"]:
        print(f"SQL Query:\n{result['query']}\n")
        print(f"Results:\n{result['results']}\n")
        print(f"Explanation:\n{result['explanation']}")
    else:
        print(f"Error: {result['error']}")