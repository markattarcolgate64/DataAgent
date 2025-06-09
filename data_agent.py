import os
from anthropic import Anthropic
from dotenv import load_dotenv
from twilio.rest import Client

# Load environment variables from .env file
load_dotenv()

class DataAgent:
    def __init__(self, api_key=None, twilio_sid=None, twilio_token=None, twilio_phone=None):
        self.model = "claude-sonnet-4-20250514"
        
        # Try to get API key from parameter or environment
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        
        if not self.api_key:
            raise ValueError("Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable or pass api_key parameter.")
        
        self.client = Anthropic(api_key=self.api_key)
        self.conversation_history = []
        self.tools = []
        
        # Twilio setup
        self.twilio_sid = twilio_sid or os.getenv("TWILIO_ACCOUNT_SID")
        self.twilio_token = twilio_token or os.getenv("TWILIO_AUTH_TOKEN")
        self.twilio_phone = twilio_phone or os.getenv("TWILIO_PHONE_NUMBER")
        
        if self.twilio_sid and self.twilio_token:
            self.twilio_client = Client(self.twilio_sid, self.twilio_token)
        else:
            self.twilio_client = None
            print("Warning: Twilio credentials not found. SMS functionality disabled.")
        
        self.set_tools()
    
    def send_message(self, message, system_prompt=None):
        """Send a message to Claude and get a response"""
        try:
            messages = self.conversation_history + [{"role": "user", "content": message}]
            
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1000,
                system=system_prompt or "You are a helpful AI assistant.",
                messages=messages,
                tools=self.tools if self.tools else None
            )
            
            assistant_message = ""

            for mssg in response.content:
                if mssg.type == "text":
                    assistant_message += mssg.text
                elif mssg.type == "tool_use":
                    tool_result = self.execute_tool(mssg.name, mssg.input)
                    assistant_message += f"\n[Tool: {mssg.name}] {tool_result}"
            
            # Update conversation history
            self.conversation_history.append({"role": "user", "content": message})
            self.conversation_history.append({"role": "assistant", "content": assistant_message})
            
            return assistant_message
            
        except Exception as e:
            return f"Error: {str(e)}"
    
    def clear_history(self):
        """Clear the conversation history"""
        self.conversation_history = []
    
    def get_response(self, prompt):
        """Simple one-off response without conversation history"""
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            return f"Error: {str(e)}"

    def execute_tool(self, tool_name, tool_input):
        if tool_name == "read_data":
            return self.read_data(tool_input)
        elif tool_name == "send_sms":
            return self.send_sms(tool_input)
        else:
            return "Tool not found"

    def read_data(self, model_input):
        data_path = model_input.get("filename")

        if data_path:
            try:
                with open(data_path, "r") as file:
                    return file.read()
            except FileNotFoundError:
                return f"File not found: {data_path}"
            except Exception as e:
                return f"Error reading file: {str(e)}"
        else:
            return "No data path provided"

    def send_sms(self, model_input):
        """Send SMS via Twilio"""
        if not self.twilio_client:
            return "Twilio not configured. Please set TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, and TWILIO_PHONE_NUMBER"
        
        to_number = model_input.get("to_number")
        message_body = model_input.get("message")
        
        if not to_number or not message_body:
            return "Missing required parameters: to_number and message"
        
        try:
            message = self.twilio_client.messages.create(
                body=message_body,
                from_=self.twilio_phone,
                to=to_number
            )
            return f"SMS sent successfully! Message SID: {message.sid}"
        except Exception as e:
            return f"Failed to send SMS: {str(e)}"

    def set_tools(self):
        """Set up available tools for the agent"""
        tools = [    
            {
                "name": "read_data",
                "description": "read data from a csv file",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "filename": {
                            "type": "string",
                            "description": "filename of the csv file to read"
                        }
                    },
                    "required": ["filename"]
                }
            }
        ]
        
        # Add SMS tool if Twilio is configured
        if self.twilio_client:
            tools.append({
                "name": "send_sms",
                "description": "send SMS text message via Twilio",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "to_number": {
                            "type": "string",
                            "description": "recipient phone number in E.164 format (e.g., +1234567890)"
                        },
                        "message": {
                            "type": "string",
                            "description": "text message content to send"
                        }
                    },
                    "required": ["to_number", "message"]
                }
            })
        
        self.tools = tools

    def interactive_chat(self, system_prompt=None):
        """Start an interactive multi-turn chat session"""
        print("Data Agent Chat")
        print("=" * 50)
        print("Commands:")
        print("  • 'quit' or 'exit' to end")
        print("  • 'clear' to reset conversation")
        print("  • 'tools' to enable data tools")
        if self.twilio_client:
            print("  • SMS functionality is enabled")
        print("=" * 50)
        
        if system_prompt:
            print(f"System: {system_prompt}")
            print("-" * 50)
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                elif user_input.lower() == 'clear':
                    self.clear_history()
                    print("Conversation history cleared!")
                    continue
                elif user_input.lower() == 'tools':
                    self.set_tools()
                    print("Data tools enabled!")
                    continue
                elif not user_input:
                    continue
                
                print("Agent: ", end="", flush=True)
                response = self.send_message(user_input, system_prompt)
                print(response)
                
            except KeyboardInterrupt:
                print("\n\nChat interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}")

# Example usage
if __name__ == "__main__":
    try:
        agent = DataAgent()
        
        # Start interactive chat
        agent.interactive_chat()
        
    except ValueError as e:
        print(f"{e}")
        print("\nTo fix this:")
        print("1. Set environment variable: export ANTHROPIC_API_KEY='your-api-key'")
        print("2. Or pass API key directly: DataAgent(api_key='your-api-key')")
        print("3. Create a .env file with: ANTHROPIC_API_KEY=your-api-key")