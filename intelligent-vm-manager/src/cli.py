#!/usr/bin/env python3
import argparse
import sys
from vm_manager import IntelligentVMManager
import os
from dotenv import load_dotenv

def main():
    # Load environment variables
    load_dotenv()
    
    parser = argparse.ArgumentParser(
        description="Intelligent VM Management Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Migrate an application using natural language
  python cli.py "migrate the sample app from Lubuntu to a more powerful VM"
  
  # Scale up a VM
  python cli.py "scale up Lubuntu to handle more load"
  
  # Get performance analysis
  python cli.py "analyze the performance of Lubuntu and suggest improvements"
        """
    )
    
    parser.add_argument(
        "command",
        help="Natural language command describing the desired VM operation"
    )
    
    parser.add_argument(
        "--api-key",
        help="OpenAI API key (defaults to OPENAI_API_KEY environment variable)",
        default=os.getenv("OPENAI_API_KEY")
    )
    
    parser.add_argument(
        "--log-file",
        help="Log file path (default: vm_operations.log)",
        default="vm_operations.log"
    )
    
    args = parser.parse_args()
    
    if not args.api_key:
        print("Error: OpenAI API key is required. Set it in .env file or use --api-key")
        sys.exit(1)
    
    try:
        vm_manager = IntelligentVMManager(
            openai_api_key=args.api_key,
            log_file=args.log_file
        )
        
        success = vm_manager.handle_migration_request(args.command)
        
        if success:
            print("Operation completed successfully")
        else:
            print("Operation failed")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 