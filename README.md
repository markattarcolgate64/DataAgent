# DataAgent

A Python-based AI agent powered by Claude that can read data files and send SMS notifications.

## Features

- Interactive chat interface with Claude Sonnet 4
- Read and process CSV files
- Send SMS messages via Twilio
- Persistent conversation history
- Tool-enabled AI interactions

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set environment variables:
```bash
export ANTHROPIC_API_KEY="your-api-key"
export TWILIO_ACCOUNT_SID="your-twilio-sid"
export TWILIO_AUTH_TOKEN="your-twilio-token"
export TWILIO_PHONE_NUMBER="your-twilio-phone"
```

Or create a `.env` file with these variables.

## Usage

Run the interactive chat:
```bash
python data_agent.py
```

### Commands
- `quit` or `exit` - End session
- `clear` - Reset conversation history
- `tools` - Enable data tools

### Programmatic Usage

```python
from data_agent import DataAgent

agent = DataAgent()
response = agent.send_message("Read the file data.csv")
```

## Requirements

- Python 3.7+
- Anthropic API key
- Twilio credentials (optional, for SMS)