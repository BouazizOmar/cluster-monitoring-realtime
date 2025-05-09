from agent import SnowflakeAIAgent

if __name__ == "__main__":
    agent = SnowflakeAIAgent()
    question = "The most used service in Lubuntu VM"
    result = agent.process_question(question)
    if result["success"]:
        print(f"SQL Query:\n{result['query']}\n")
        print(f"Results:\n{result['results']}\n")
        print(f"Explanation:\n{result['explanation']}")
    else:
        print(f"Error: {result['error']}")