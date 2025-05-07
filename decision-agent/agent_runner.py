# agent_runner.py
import logging
import os

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain.agents import initialize_agent, AgentType

from tools import get_new_prompt, infer_action

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pipeline_agent")

# your fine-grained HF token with gated-repo access
HF_TOKEN = os.environ["HUGGINGFACE_HUB_TOKEN"]

def build_agent():
    # 1) Register your two tools
    tools = [get_new_prompt, infer_action]

    # 2) Load the chat-capable LLM (Mistral 7B Instruct here)
    model_name = "mistralai/Mistral-7B-Instruct-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        token=HF_TOKEN,
        local_files_only=False
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype="auto",
        offload_folder="offload",
        offload_state_dict=True,
        token=HF_TOKEN
    )

    hf_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=150,
        do_sample=False
    )
    llm = HuggingFacePipeline(pipeline=hf_pipe)

    # 3) Initialize the **chat** ReAct agent
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,  # ← chat-based agent
        verbose=True,
    )
    return agent

if __name__ == "__main__":
    agent = build_agent()
    # natural-language question—agent will pick get_new_prompt(), then infer_action()
    question = "Fetch the latest monitoring report and recommend actions."
    result = agent.run(question)
    print("\n=== Final Decision ===\n", result)
