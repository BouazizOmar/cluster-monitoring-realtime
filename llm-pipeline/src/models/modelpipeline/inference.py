import os
from datetime import datetime
from transformers import AutoTokenizer
from mlx_lm import load, generate as mlx_generate

# Simple logging function
def log(message, level="INFO"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
    print(f"{timestamp} - inference - {level} - {message}")

def run_inference(
    model_path="mlx-community/Mistral-7B-Instruct-v0.3-4bit",
    adapter_path="adapters",
    prompt="",
    output_file=None,
    max_tokens=512,
    verbose=False
):
    """Simple function to run inference with a fine-tuned model."""
    # Load tokenizer
    log(f"Loading tokenizer from modified_tokenizer")
    tokenizer = AutoTokenizer.from_pretrained("modified_tokenizer", use_fast=False)
    
    # Load model with adapters
    log(f"Loading model from {model_path} with adapters from {adapter_path}")
    model, _ = load(model_path, adapter_path=adapter_path)
    model.eval()
    
    # Generate text
    if verbose:
        log(f"Prompt: {prompt}")
    
    log("Starting text generation")
    generated_text = mlx_generate(
        model, 
        tokenizer, 
        prompt=prompt,
        max_tokens=max_tokens,
    )
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write(f"Prompt:\n{prompt}\n\nResponse:\n{generated_text}")
        log(f"Results saved to {output_file}")
    
    return generated_text


VM_STATUS_PROMPT = """
You are a cloud operations specialist. You will be given a system monitoring report for multiple virtual machines. For each VM:

  1. Compare its metrics against healthy baselines.
  2. Identify any anomalies.
  3. Choose the single most appropriate action from:
     - NO_ACTION
     - RESTART_SERVICES
     - RESTART_VM
     - MIGRATE_VM
     - SCALE_UP
  4. Explain your reasoning in 1–2 sentences.

Report format:

---
VM Name: <string>
CPU Idle Time: <seconds>
Memory Used: <percent> of <total GB> (Available: <GB>)
Disk I/O Time: <seconds>
Failed Services (Count: <n>): [<service1>, <service2>, …]
Detected Anomalies: <anomaly_codes>
---

Your output must follow this template exactly:

---
VM Name: <name>  
Recommended Action: <one of the five options>  
Reason: <concise explanation>  
---

System Monitoring Report (2025‑04‑06 16:34:00 UTC):

VM Lubuntu:
  - CPU Idle: 986.9 seconds  
  - Memory used: 21.8% of 3.7GB (Available: 2.9GB)  
  - Disk I/O time: 50.6 seconds  
  - Failed services: ['NetworkManager.service', 'systemd-journald.service', 'systemd-logind.service'] (Count: 3)  

VM Lubuntu V2:
  - CPU Idle: 975.8 seconds  
  - Memory used: 21.5% of 3.7GB (Available: 2.9GB)  
  - Disk I/O time: 24.0 seconds  
  - Failed services: ['systemd-journald.service', 'systemd-logind.service', 'NetworkManager.service'] (Count: 3)  

VM Ubuntu:
  - CPU Idle: 917.9 seconds  
  - Memory used: 23.5% of 6.7GB (Available: 5.1GB)  
  - Disk I/O time: 38.5 seconds  
  - Failed services: ['NetworkManager.service', 'nginx.service', 'systemd-journald.service', '…'] (Count: 4)  

Now analyze and recommend.
"""

if __name__ == "__main__":
    # Run inference with the example prompt
    response = run_inference(
        prompt=VM_STATUS_PROMPT,
        output_file="vm_analysis_result.json",
        verbose=True
    )
    
    print("\n===== GENERATED RESPONSE =====")
    print(response)
    print("=============================\n")