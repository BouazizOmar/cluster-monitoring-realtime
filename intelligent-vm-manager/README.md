# Intelligent VM Manager

An intelligent VirtualBox VM management tool that uses OpenAI's GPT models to understand natural language commands and perform VM operations like migration and scaling.

## Features

- Natural language command processing
- Intelligent VM selection for migrations
- Automatic backup creation before operations
- Detailed logging of all operations
- Performance analysis and recommendations
- Resource scaling capabilities

## Prerequisites

- Python 3.7+
- VirtualBox installed and configured
- OpenAI API key

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd intelligent-vm-manager
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root:
```
OPENAI_API_KEY=your-api-key-here
```

## Usage

The tool can be used with natural language commands:

```bash
# Migrate an application
python src/cli.py "migrate the sample app from Lubuntu to a more powerful VM"

# Scale up a VM
python src/cli.py "scale up Lubuntu to handle more load"

# Get performance analysis
python src/cli.py "analyze the performance of Lubuntu and suggest improvements"
```

### Command Line Options

- `command`: The natural language command describing the desired operation
- `--api-key`: OpenAI API key (defaults to OPENAI_API_KEY environment variable)
- `--log-file`: Log file path (default: vm_operations.log)

## Project Structure

```
intelligent-vm-manager/
├── src/
│   ├── vm_manager.py    # Core VM management functionality
│   └── cli.py          # Command-line interface
├── logs/               # Log files directory
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Logging

Logs are stored in the `logs` directory. Each operation is logged with timestamps and detailed information about the actions performed.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
