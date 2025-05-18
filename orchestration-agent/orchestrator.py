import os
import json
import logging
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional, List, TypedDict
from enum import Enum
from dataclasses import dataclass
import subprocess

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import Graph, END
from pydantic import BaseModel, Field

load_dotenv()

class DecisionType(str, Enum):
    CRITICAL_MIGRATION = "critical_migration"
    CRITICAL_SCALING = "critical_scaling"
    CRITICAL_DEPLOYMENT = "critical_deployment"
    STANDARD_MONITORING = "standard_monitoring"
    CONFIGURATION = "configuration"
    INFORMATION = "information"

class DecisionInput(BaseModel):
    content: str = Field(description="The decision content to process")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    source: str = Field(default="api", description="Source of the decision")

class ClassifiedDecision(BaseModel):
    id: str
    type: DecisionType
    content: str
    priority: int = Field(ge=1, le=10)
    metadata: Dict[str, Any]
    timestamp: datetime
    reasoning: str
    formatted_for_vm: Optional[Dict[str, Any]] = None

class GraphState(TypedDict):
    decision_input: DecisionInput
    classified_decision: Optional[ClassifiedDecision]
    vm_manager_result: Optional[Dict[str, Any]]
    final_result: Optional[Dict[str, Any]]
    error: Optional[str]

class DecisionClassifier:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.classification_prompt = PromptTemplate(
            input_variables=["content", "metadata"],
            template="""
            You are an expert system administrator analyzing operational decisions.
            
            Decision Content: {content}
            Metadata: {metadata}
            
            Classify this decision and extract relevant parameters. Respond ONLY with valid JSON:
            
            Categories:
            1. critical_migration - Moving applications between VMs
            2. critical_scaling - Scaling up/down VMs or resources
            3. critical_deployment - Deploying or updating applications
            4. standard_monitoring - Regular monitoring tasks
            5. configuration - Configuration changes
            6. information - Informational queries
            
            Response format (MUST be valid JSON):
            {{
                "type": "decision_type",
                "priority": 5,
                "reasoning": "explanation of classification",
                "vm_params": {{
                    "vm_name": "vm_name_if_specified",
                    "cpus": 2,
                    "memory": 1024,
                    "scale_action": "scale_up/scale_down",
                    "vm_count": 2,
                    "target_vm": "vm_id_if_applicable",
                    "source_vm": "vm_id_if_applicable",
                    "application": "app_name_if_applicable",
                    "app_name": "application_name",
                    "operation": "operation_type"
                }}
            }}
            
            For scaling operations, extract:
            - vm_name: Specific VM identifier from content (e.g., "ubuntu-24.10", "Lubuntu_V2")
            - cpus: Number of CPUs (default: 2)
            - memory: Memory in MB (default: 1024)
            - vm_count: Number of instances to scale
            
            IMPORTANT: Do not use generic names like "web-server" or "web_server". 
            If no specific VM is mentioned, leave vm_name empty so it can be determined later.
            """
        )
    
    async def classify(self, content: str, metadata: Dict[str, Any]) -> ClassifiedDecision:
        """Classify decision using LLM"""
        try:
            # Create system message for JSON response
            system_msg = SystemMessage(content="""You are a precise decision classifier for VM operations. 
            You MUST respond with valid JSON only. No explanations, no markdown, just pure JSON.
            Extract specific VM parameters like vm_name, cpus, memory from the decision content.
            Only use actual VM names mentioned in the content, not generic names.""")
            
            # Create human message with the prompt
            prompt = self.classification_prompt.format(
                content=content,
                metadata=json.dumps(metadata, indent=2)
            )
            human_msg = HumanMessage(content=prompt)
            
            # Get response from LLM
            response = await self.llm.ainvoke([system_msg, human_msg])
            
            # Clean the response content (remove any markdown if present)
            content_str = response.content.strip()
            if content_str.startswith("```json"):
                content_str = content_str.split("```json")[1].split("```")[0].strip()
            elif content_str.startswith("```"):
                content_str = content_str.split("```")[1].split("```")[0].strip()
            
            # Parse JSON response
            try:
                result = json.loads(content_str)
            except json.JSONDecodeError as e:
                logging.error(f"Invalid JSON response: {content_str}")
                raise e
            
            # Create classified decision
            decision_id = f"dec_{int(datetime.now().timestamp() * 1000)}"
            
            # Validate decision type
            decision_type = result.get("type", "information")
            if decision_type not in [dt.value for dt in DecisionType]:
                decision_type = "information"
            
            # Extract VM parameters with defaults
            vm_params = result.get("vm_params", {})
            formatted_for_vm = None
            
            if decision_type.startswith("critical_"):
                # Use metadata vm_name if provided, otherwise keep LLM result
                if "vm_name" in metadata:
                    vm_params["vm_name"] = metadata["vm_name"]
                # Don't set a default vm_name - let the VM manager determine it
                
                vm_params.setdefault("cpus", 2)
                vm_params.setdefault("memory", 1024)
                vm_params.setdefault("vm_count", 1)
                
                formatted_for_vm = {
                    "action": decision_type,
                    "content": content,
                    "priority": result.get("priority", 5),
                    "timestamp": datetime.now().isoformat(),
                    "metadata": metadata,
                    **vm_params
                }
            
            return ClassifiedDecision(
                id=decision_id,
                type=DecisionType(decision_type),
                content=content,
                priority=result.get("priority", 5),
                metadata=metadata,
                timestamp=datetime.now(),
                reasoning=result.get("reasoning", "Classification completed"),
                formatted_for_vm=formatted_for_vm
            )
            
        except Exception as e:
            logging.error(f"Classification error: {e}")
            # Enhanced fallback classification with keyword matching
            decision_type = self._fallback_classify(content)
            
            # Extract basic parameters from content for fallback
            vm_params = self._extract_fallback_params(content, metadata)
            
            formatted_for_vm = None
            if decision_type.value.startswith("critical_"):
                formatted_for_vm = {
                    "action": decision_type.value,
                    "content": content,
                    "priority": 8,
                    "timestamp": datetime.now().isoformat(),
                    "metadata": metadata,
                    **vm_params
                }
            
            return ClassifiedDecision(
                id=f"dec_{int(datetime.now().timestamp() * 1000)}",
                type=decision_type,
                content=content,
                priority=8 if decision_type.value.startswith("critical_") else 5,
                metadata=metadata,
                timestamp=datetime.now(),
                reasoning=f"Fallback classification due to error: {str(e)}",
                formatted_for_vm=formatted_for_vm
            )
    
    def _fallback_classify(self, content: str) -> DecisionType:
        """Fallback classification using keywords"""
        content_lower = content.lower()
        
        # Critical operation keywords
        if any(word in content_lower for word in ["migrate", "migration", "move app"]):
            return DecisionType.CRITICAL_MIGRATION
        elif any(word in content_lower for word in ["scale up", "scale down", "instances", "VM count"]):
            return DecisionType.CRITICAL_SCALING
        elif any(word in content_lower for word in ["deploy", "deployment", "rollback", "update app"]):
            return DecisionType.CRITICAL_DEPLOYMENT
        elif any(word in content_lower for word in ["monitor", "check", "status"]):
            return DecisionType.STANDARD_MONITORING
        elif any(word in content_lower for word in ["config", "configure", "settings"]):
            return DecisionType.CONFIGURATION
        else:
            return DecisionType.INFORMATION
    
    def _extract_fallback_params(self, content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Extract basic parameters for fallback classification"""
        params = {
            "cpus": metadata.get("cpus", 2),
            "memory": metadata.get("memory", 1024),
            "vm_count": metadata.get("vm_count", 1)
        }
        
        # Use vm_name from metadata if provided
        if "vm_name" in metadata:
            params["vm_name"] = metadata["vm_name"]
        
        # Try to extract vm_count from content
        import re
        numbers = re.findall(r'\d+', content.lower())
        if numbers:
            # Look for phrases like "by 2" or "2 instances"
            if "by" in content.lower() or "instances" in content.lower():
                params["vm_count"] = int(numbers[-1])
        
        return params

class VMManagerInterface:
    def __init__(self, vm_manager_path: str):
        self.vm_manager_path = os.path.abspath(vm_manager_path)
        self.logger = logging.getLogger(__name__)
        
        # Check if VM manager has its own venv
        self.vm_manager_venv = os.path.join(self.vm_manager_path, "venv")
        self.has_venv = os.path.exists(self.vm_manager_venv)
        
        if self.has_venv:
            self.python_path = os.path.join(self.vm_manager_venv, "bin", "python")
        else:
            self.python_path = "python"
        
        self.logger.info(f"VM Manager path: {self.vm_manager_path}")
        self.logger.info(f"Using Python: {self.python_path}")
    
    async def get_available_vms(self) -> List[str]:
        """Get list of available VMs"""
        try:
            cmd = ["VBoxManage", "list", "vms"]
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                self.logger.error(f"Failed to list VMs: {stderr.decode()}")
                return []
            
            # Parse VM names from output
            vm_names = []
            for line in stdout.decode().strip().split('\n'):
                if line and '"' in line:
                    # Extract VM name from quotes
                    vm_name = line.split('"')[1]
                    vm_names.append(vm_name)
            
            return vm_names
        except Exception as e:
            self.logger.error(f"Error getting VMs: {e}")
            return []
    
    def _build_vm_autoscaler_command(self, decision: ClassifiedDecision) -> List[str]:
        """Build command for vm_autoscaler.py based on decision parameters"""
        params = decision.formatted_for_vm
        
        # Extract required parameters
        vm_name = params.get("vm_name")
        cpus = params.get("cpus", 2)
        memory = params.get("memory", 1024)
        
        # If no VM name specified, get the first available VM
        if not vm_name:
            # For now, use a placeholder - the actual implementation would need to be async
            # You could modify this to use a default VM or require vm_name in metadata
            vm_name = "ubuntu-24.10"  # Use one of your existing VMs as default
            self.logger.warning(f"No VM name specified, using default: {vm_name}")
        
        # Build basic command
        cmd = [
            self.python_path,
            os.path.join(self.vm_manager_path, "vm_autoscaler.py"),
            vm_name,
            "--cpus", str(cpus),
            "--memory", str(memory)
        ]
        
        # Add optional parameters
        if params.get("app_name"):
            cmd.extend(["--app-name", params["app_name"]])
        
        if params.get("health_url"):
            cmd.extend(["--health-url", params["health_url"]])
        
        # Add debug flag for better logging
        cmd.append("--debug")
        
        return cmd
    
    def _build_app_migrator_command(self, decision: ClassifiedDecision) -> List[str]:
        """Build command for app_migrator.py based on decision parameters"""
        params = decision.formatted_for_vm
        
        # Extract required parameters
        source_vm = params.get("source_vm", "")
        target_vm = params.get("target_vm", "")
        application = params.get("application", "")
        
        if not source_vm or not target_vm or not application:
            raise ValueError("Missing required parameters for migration")
        
        cmd = [
            self.python_path,
            os.path.join(self.vm_manager_path, "app_migrator.py"),
            "--source-vm", source_vm,
            "--target-vm", target_vm,
            "--application", application
        ]
        
        # Add optional parameters
        if params.get("port"):
            cmd.extend(["--port", str(params["port"])])
        
        if params.get("health_url"):
            cmd.extend(["--health-url", params["health_url"]])
        
        if params.get("timeout"):
            cmd.extend(["--timeout", str(params["timeout"])])
        
        # Add debug flag for better logging
        cmd.append("--debug")
        
        return cmd
    
    async def execute_critical_operation(self, decision: ClassifiedDecision) -> Dict[str, Any]:
        """Execute critical operation via VM manager with proper arguments"""
        try:
            # Build command based on operation type
            if decision.type == DecisionType.CRITICAL_MIGRATION:
                cmd = self._build_app_migrator_command(decision)
            else:  # CRITICAL_SCALING, CRITICAL_DEPLOYMENT
                cmd = self._build_vm_autoscaler_command(decision)
            
            # Check if script exists
            script_path = cmd[1]
            if not os.path.exists(script_path):
                return {
                    "success": False,
                    "error": f"Script not found: {script_path}"
                }
            
            # Set up environment
            if self.has_venv:
                env = os.environ.copy()
                env["PATH"] = f"{os.path.join(self.vm_manager_venv, 'bin')}:{env.get('PATH', '')}"
                env["VIRTUAL_ENV"] = self.vm_manager_venv
            else:
                # Ensure VM manager has its environment set up
                await self._setup_vm_manager_env()
                env = os.environ.copy()
            
            self.logger.info(f"Executing: {' '.join(cmd)}")
            
            # Execute asynchronously with proper environment
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=self.vm_manager_path
            )
            
            stdout, stderr = await process.communicate()
            
            return {
                "success": process.returncode == 0,
                "stdout": stdout.decode(),
                "stderr": stderr.decode(),
                "return_code": process.returncode,
                "command": ' '.join(cmd)
            }
            
        except Exception as e:
            self.logger.error(f"VM manager execution error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _setup_vm_manager_env(self):
        """Setup VM manager's environment if it doesn't exist"""
        if self.has_venv:
            return
        
        try:
            # Create virtual environment for VM manager
            self.logger.info("Setting up VM manager environment...")
            
            # Create venv
            process = await asyncio.create_subprocess_exec(
                "python3", "-m", "venv", self.vm_manager_venv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.vm_manager_path
            )
            await process.communicate()
            
            # Install requirements
            requirements_file = os.path.join(self.vm_manager_path, "requirements.txt")
            if os.path.exists(requirements_file):
                pip_path = os.path.join(self.vm_manager_venv, "bin", "pip")
                process = await asyncio.create_subprocess_exec(
                    pip_path, "install", "-r", requirements_file,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=self.vm_manager_path
                )
                await process.communicate()
            
            self.has_venv = True
            self.python_path = os.path.join(self.vm_manager_venv, "bin", "python")
            self.logger.info("VM manager environment setup complete")
            
        except Exception as e:
            self.logger.error(f"Failed to setup VM manager environment: {e}")

    async def check_vm_resources(self, vm_name: str) -> Dict[str, Any]:
        """Check current resource state of a VM"""
        try:
            cmd = [
                self.python_path,
                os.path.join(self.vm_manager_path, "vm_resource_checker.py"),
                vm_name,
                "--json"
            ]
            
            # Set up environment
            if self.has_venv:
                env = os.environ.copy()
                env["PATH"] = f"{os.path.join(self.vm_manager_venv, 'bin')}:{env.get('PATH', '')}"
                env["VIRTUAL_ENV"] = self.vm_manager_venv
            else:
                env = os.environ.copy()
            
            self.logger.info(f"Checking resources for VM: {vm_name}")
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=self.vm_manager_path
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                return {
                    "success": False,
                    "error": stderr.decode(),
                    "vm_name": vm_name
                }
            
            try:
                result = json.loads(stdout.decode())
                return {
                    "success": True,
                    "vm_name": vm_name,
                    "resource_state": result
                }
            except json.JSONDecodeError as e:
                return {
                    "success": False,
                    "error": f"Failed to parse resource state: {e}",
                    "vm_name": vm_name
                }
            
        except Exception as e:
            self.logger.error(f"Error checking VM resources: {e}")
            return {
                "success": False,
                "error": str(e),
                "vm_name": vm_name
            }

class OrchestrationAgent:
    def __init__(self, openai_api_key: str = None):
        self.logger = self._setup_logging()
        
        # Initialize LLM with explicit JSON mode request
        self.llm = ChatOpenAI(
            api_key=openai_api_key or os.getenv("OPENAI_API_KEY"),
            model="gpt-3.5-turbo",  # Changed to 3.5-turbo for better JSON compliance
            temperature=0.1,
            model_kwargs={"response_format": {"type": "json_object"}}
        )
        
        # Initialize components
        self.classifier = DecisionClassifier(self.llm)
        self.vm_manager = VMManagerInterface(
            os.getenv("VM_MANAGER_PATH", "../intelligent-vm-manager")
        )
        
        # Create LangGraph workflow
        self.graph = self._create_workflow()
        
    def _setup_logging(self) -> logging.Logger:
        logging.basicConfig(
            level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('orchestrator.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    def _create_workflow(self) -> Graph:
        """Create LangGraph workflow for decision processing"""
        workflow = Graph()
        
        # Add nodes
        workflow.add_node("classify", self._classify_node)
        workflow.add_node("evaluate_criticality", self._evaluate_criticality_node)
        workflow.add_node("execute_vm_operation", self._execute_vm_operation_node)
        workflow.add_node("handle_standard_operation", self._handle_standard_operation_node)
        workflow.add_node("finalize", self._finalize_node)
        
        # Add edges
        workflow.add_edge("classify", "evaluate_criticality")
        workflow.add_conditional_edges(
            "evaluate_criticality",
            self._should_use_vm_manager,
            {
                "critical": "execute_vm_operation",
                "standard": "handle_standard_operation"
            }
        )
        workflow.add_edge("execute_vm_operation", "finalize")
        workflow.add_edge("handle_standard_operation", "finalize")
        workflow.add_edge("finalize", END)
        
        # Set entry point
        workflow.set_entry_point("classify")
        
        return workflow.compile()
    
    async def _classify_node(self, state: GraphState) -> GraphState:
        """Classify the incoming decision"""
        try:
            decision_input = state["decision_input"]
            classified = await self.classifier.classify(
                decision_input.content, 
                decision_input.metadata
            )
            state["classified_decision"] = classified
            self.logger.info(f"Classified decision {classified.id} as {classified.type}")
        except Exception as e:
            state["error"] = f"Classification error: {str(e)}"
            self.logger.error(state["error"])
        return state
    
    async def _evaluate_criticality_node(self, state: GraphState) -> GraphState:
        """Evaluate if decision needs VM manager intervention"""
        decision = state["classified_decision"]
        if decision and decision.type.value.startswith("critical_"):
            # Add safety delay for critical operations
            delay = 3  # seconds
            self.logger.info(f"Critical operation detected. Waiting {delay}s...")
            await asyncio.sleep(delay)
        return state
    
    async def _execute_vm_operation_node(self, state: GraphState) -> GraphState:
        """Execute operation via VM manager"""
        try:
            decision = state["classified_decision"]
            if decision:
                result = await self.vm_manager.execute_critical_operation(decision)
                state["vm_manager_result"] = result
                self.logger.info(f"VM operation result: {result.get('success', False)}")
        except Exception as e:
            state["error"] = f"VM execution error: {str(e)}"
            self.logger.error(state["error"])
        return state
    
    async def _handle_standard_operation_node(self, state: GraphState) -> GraphState:
        """Handle non-critical operations"""
        decision = state["classified_decision"]
        if decision:
            # Handle resource check operations
            if decision.type == DecisionType.STANDARD_MONITORING:
                # Extract VM name from content or metadata
                vm_name = None
                if decision.formatted_for_vm and "vm_name" in decision.formatted_for_vm:
                    vm_name = decision.formatted_for_vm["vm_name"]
                elif "vm_name" in decision.metadata:
                    vm_name = decision.metadata["vm_name"]
                
                if vm_name:
                    # Check VM resources
                    result = await self.vm_manager.check_vm_resources(vm_name)
                    state["vm_manager_result"] = result
                    self.logger.info(f"Resource check completed for VM: {vm_name}")
                else:
                    state["vm_manager_result"] = {
                        "success": False,
                        "error": "No VM name specified for resource check",
                        "type": decision.type.value
                    }
            else:
                # Handle other standard operations
                result = {
                    "success": True,
                    "message": f"Standard operation handled: {decision.content}",
                    "type": decision.type.value
                }
                state["vm_manager_result"] = result
                self.logger.info(f"Standard operation completed: {decision.id}")
        return state
    
    async def _finalize_node(self, state: GraphState) -> GraphState:
        """Finalize the decision processing"""
        decision = state["classified_decision"]
        vm_result = state.get("vm_manager_result", {})
        error = state.get("error")
        
        final_result = {
            "decision_id": decision.id if decision else "unknown",
            "decision_type": decision.type.value if decision else "unknown",
            "priority": decision.priority if decision else 0,
            "success": vm_result.get("success", False) if not error else False,
            "reasoning": decision.reasoning if decision else "",
            "timestamp": datetime.now().isoformat()
        }
        
        if error:
            final_result["error"] = error
        if vm_result:
            final_result["execution_result"] = vm_result
        
        state["final_result"] = final_result
        return state
    
    def _should_use_vm_manager(self, state: GraphState) -> str:
        """Conditional edge: decide whether to use VM manager"""
        decision = state.get("classified_decision")
        if decision and decision.type.value.startswith("critical_"):
            return "critical"
        return "standard"
    
    async def process_decision(self, decision_input: DecisionInput) -> Dict[str, Any]:
        """Process a decision through the workflow"""
        initial_state: GraphState = {
            "decision_input": decision_input,
            "classified_decision": None,
            "vm_manager_result": None,
            "final_result": None,
            "error": None
        }
        
        try:
            # Run the workflow
            final_state = await self.graph.ainvoke(initial_state)
            return final_state["final_result"]
        except Exception as e:
            self.logger.error(f"Workflow execution error: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

# API Server using aiohttp
from aiohttp import web, web_request

class OrchestrationAPI:
    def __init__(self, agent: OrchestrationAgent):
        self.agent = agent
        self.app = web.Application()
        self._setup_routes()
    
    def _setup_routes(self):
        self.app.router.add_post('/decision', self.submit_decision)
        self.app.router.add_get('/health', self.health_check)
        self.app.router.add_get('/status', self.get_status)
    
    async def submit_decision(self, request: web_request.Request):
        try:
            data = await request.json()
            decision_input = DecisionInput(**data)
            result = await self.agent.process_decision(decision_input)
            return web.json_response(result)
        except Exception as e:
            return web.json_response({"success": False, "error": str(e)}, status=400)
    
    async def health_check(self, request: web_request.Request):
        return web.json_response({"status": "healthy"})
    
    async def get_status(self, request: web_request.Request):
        return web.json_response({"status": "running"})
    
    async def start_server(self, host='0.0.0.0', port=8082):
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, host, port)
        await site.start()
        print(f"Orchestration API running on http://{host}:{port}")

# Main execution
async def main():
    # Initialize agent
    agent = OrchestrationAgent()
    
    # Start API server
    api = OrchestrationAPI(agent)
    await api.start_server()
    
    # Keep running
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down...")

if __name__ == "__main__":
    asyncio.run(main())