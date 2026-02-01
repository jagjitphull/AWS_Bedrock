# AWS Bedrock Agent - Complete Step-by-Step Guide

## Use Case: Weather Information Assistant

We'll build a simple Bedrock agent that can fetch weather information for cities using an API. This agent will demonstrate all core concepts of Bedrock Agents.

---

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Architecture Overview](#architecture-overview)
3. [Step 1: Create Your First Bedrock Agent](#step-1-create-your-first-bedrock-agent)
4. [Step 2: Define Instructions and Capabilities](#step-2-define-instructions-and-capabilities)
5. [Step 3: Create Action Groups with API Schemas](#step-3-create-action-groups-with-api-schemas)
6. [Step 4: Invoke Agent (Console, SDK, CLI)](#step-4-invoke-agent-console-sdk-cli)
7. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required AWS Services Access
- AWS Bedrock (with Claude or other foundation models enabled)
- AWS Lambda
- IAM permissions to create/manage agents
- AWS CLI configured (for CLI invocation)

### Tools Needed
- AWS Console access
- AWS CLI installed and configured
- Python 3.9+ (for SDK examples)
- boto3 library: `pip install boto3`

### Foundation Model Access
Enable at least one foundation model in Bedrock:
- Go to AWS Bedrock Console → Model access
- Request access to Claude 3 Sonnet or Haiku (recommended for this tutorial)
- Wait for approval (usually instant for most models)

---

## Architecture Overview

```
User Query → Bedrock Agent → Action Group → Lambda Function → External API
                ↓
           Foundation Model (Claude)
                ↓
           Response to User
```

**Components:**
- **Agent (AgentCore)**: Orchestrates the conversation and decision-making
- **Instructions**: Guide the agent's behavior and personality
- **Action Groups**: Define what actions the agent can perform
- **Lambda Functions**: Execute the actual API calls
- **OpenAPI Schema**: Describes the API structure to the agent

---

## Step 1: Create Your First Bedrock Agent

### 1.1 Navigate to Bedrock Console

1. Open AWS Console
2. Search for "Bedrock" and select it
3. In the left sidebar, click **Agents** → **Create Agent**

### 1.2 Configure Agent Details

**Agent Name:** `WeatherAssistant`

**Description:** `An AI agent that provides weather information for cities worldwide`

**User Input:** Leave "Enable" selected (allows users to interact with the agent)

### 1.3 Select Foundation Model

Choose your foundation model:
- **Model:** `Anthropic Claude 3 Sonnet` (recommended)
- **Why?** Claude models excel at understanding context and following instructions

Click **Next**

### 1.4 Create IAM Service Role

**Option 1 - Let AWS Create Role (Recommended for beginners)**
- Select "Create and use a new service role"
- Role name: `AmazonBedrockExecutionRoleForAgents_WeatherAssistant`

**Option 2 - Use Existing Role**
- The role needs these permissions:
  - `AmazonBedrockFullAccess`
  - Permission to invoke Lambda functions
  - CloudWatch Logs permissions

### Explanation
The service role allows Bedrock to:
- Invoke the foundation model
- Call Lambda functions in your action groups
- Write logs to CloudWatch
- Access other AWS resources as needed

---

## Step 2: Define Instructions and Capabilities

### 2.1 Write Agent Instructions

**Where to Enter Instructions:**

After creating your agent in Step 1, you'll be on the agent configuration page:

1. **Scroll down** to the section titled **"Instructions for the Agent"** 
2. You'll see a large text box with placeholder text
3. **Click inside the text box** to start editing
4. **Delete** any default text
5. **Paste or type** your custom instructions (see below)
6. **Optional**: Click **"Show example"** link to see AWS-provided examples

**Location in Console Flow:**
```
Create Agent → Agent Details → [Scroll Down] → Instructions for the Agent (text box)
```

**Console Layout (what you'll see):**
```
┌─────────────────────────────────────────────────────┐
│ Agent details                                        │
│ ✓ Agent name: WeatherAssistant                      │
│ ✓ Foundation model: Claude 3 Sonnet                 │
│ ✓ Service role: AmazonBedrockExecutionRole...       │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│ Instructions for the Agent - optional    [Show example]│
│                                                      │
│ ┌───────────────────────────────────────────────┐  │
│ │ You are a helpful weather information         │  │
│ │ assistant. Your role is to:                   │  │
│ │                                                │  │
│ │ [Enter your instructions here...]             │  │
│ │                                                │  │
│ │                                                │  │
│ └───────────────────────────────────────────────┘  │
│                                        0/4000 chars │
└─────────────────────────────────────────────────────┘

[Additional settings section will appear below...]
```

Instructions are the "system prompt" that guides your agent's behavior. They define personality, capabilities, and limitations.

**Example Instructions to Enter:**

```text
You are a helpful weather information assistant. Your role is to:

1. Provide current weather information for any city the user asks about
2. Be conversational and friendly in your responses
3. If a user asks about weather, use the GetWeather action to fetch real-time data
4. Summarize weather information in an easy-to-understand format
5. If you cannot find weather data for a location, politely inform the user

Guidelines:
- Always greet users warmly
- Confirm the city name if there's any ambiguity
- Present temperature, conditions, and humidity when available
- Use appropriate units (offer both Celsius and Fahrenheit if relevant)
- Do not make up weather information - only use data from the GetWeather action

Example interaction:
User: "What's the weather in Paris?"
You: "Let me check the current weather in Paris for you."
[Call GetWeather action]
You: "In Paris, it's currently 18°C (64°F) with partly cloudy skies and 65% humidity. Perfect weather for a stroll!"
```

### 2.2 Understanding Instructions

**What the Console Shows:**
- Section header: "Instructions for the Agent - optional"
- Subtitle: "Provide instructions to help guide the agent's behavior"
- Large multi-line text box (expandable)
- Character count display
- "Show example" link in the top right

**Important Notes:**
- Instructions are **optional** but highly recommended
- You can enter up to **4000 characters**
- Instructions can be updated anytime (remember to "Prepare" agent after changes)
- More specific instructions = more predictable agent behavior

**Key Components:**

1. **Role Definition**: "You are a helpful weather information assistant"
   - Sets the agent's identity and purpose

2. **Capabilities**: What the agent can do
   - Guides when to use action groups
   - Defines scope of assistance

3. **Behavioral Guidelines**: How the agent should act
   - Tone and personality
   - Response format
   - Error handling

4. **Examples**: Show desired behavior
   - Helps the model understand expected flow
   - Demonstrates action group usage

### 2.3 Advanced Instructions (Optional)

You can add more sophisticated instructions:

```text
Advanced behaviors:
- If asked about weather for multiple cities, process them in sequence
- If asked about future weather, explain you only have current conditions
- Convert between temperature units when requested
- Suggest appropriate clothing based on conditions
```

### 2.4 Configure Additional Settings

**Idle session timeout:** 600 seconds (10 minutes)
- How long before an inactive session expires

**Enable user confirmation:** No (for this simple use case)
- When enabled, agent asks user to confirm before executing actions

**Prepare agent automatically:** Yes
- Automatically creates agent versions after changes

---

## Step 3: Create Action Groups with API Schemas

Action groups connect your agent to actual functionality through Lambda functions and API schemas.

### 3.1 Create the Lambda Function First

Before creating the action group, we need a Lambda function.

**Step 3.1.1: Create Lambda Function**

1. Go to AWS Lambda Console
2. Click **Create function**
3. Select **Author from scratch**
4. Configuration:
   - **Function name:** `WeatherAgentFunction`
   - **Runtime:** Python 3.12
   - **Architecture:** x86_64

**Lambda Function Code:**

```python
import json
import random

def lambda_handler(event, context):
    """
    Lambda function to handle weather requests from Bedrock Agent
    
    Event structure from Bedrock Agent:
    {
        "actionGroup": "WeatherActionGroup",
        "apiPath": "/weather",
        "httpMethod": "GET",
        "parameters": [
            {
                "name": "city",
                "type": "string",
                "value": "Paris"
            }
        ]
    }
    """
    
    print(f"Received event: {json.dumps(event)}")
    
    # Extract the action group and API path
    action_group = event.get('actionGroup', '')
    api_path = event.get('apiPath', '')
    
    # Extract parameters
    parameters = {param['name']: param['value'] 
                  for param in event.get('parameters', [])}
    
    city = parameters.get('city', 'Unknown')
    
    # Simulate weather data (in production, call real weather API)
    weather_data = get_mock_weather(city)
    
    # Format response for Bedrock Agent
    response = {
        'messageVersion': '1.0',
        'response': {
            'actionGroup': action_group,
            'apiPath': api_path,
            'httpMethod': 'GET',
            'httpStatusCode': 200,
            'responseBody': {
                'application/json': {
                    'body': json.dumps(weather_data)
                }
            }
        }
    }
    
    print(f"Sending response: {json.dumps(response)}")
    return response


def get_mock_weather(city):
    """
    Simulates weather API response
    In production, replace with actual API call (OpenWeatherMap, WeatherAPI, etc.)
    """
    
    # Mock weather conditions
    conditions = ["Sunny", "Partly Cloudy", "Cloudy", "Rainy", "Stormy"]
    
    weather = {
        "city": city,
        "temperature_celsius": round(random.uniform(10, 35), 1),
        "temperature_fahrenheit": 0,  # Will calculate
        "condition": random.choice(conditions),
        "humidity": random.randint(30, 90),
        "wind_speed_kmh": round(random.uniform(5, 40), 1)
    }
    
    # Calculate Fahrenheit
    weather["temperature_fahrenheit"] = round(
        (weather["temperature_celsius"] * 9/5) + 32, 1
    )
    
    return weather
```

**Step 3.1.2: Test Lambda Function**

Create a test event:

```json
{
  "actionGroup": "WeatherActionGroup",
  "apiPath": "/weather",
  "httpMethod": "GET",
  "parameters": [
    {
      "name": "city",
      "type": "string",
      "value": "London"
    }
  ]
}
```

Click **Test** - you should see a successful response with weather data.

### 3.2 Create the OpenAPI Schema

The OpenAPI schema tells Bedrock Agent what actions are available and how to use them.

**Create file: `weather-api-schema.json`**

```json
{
  "openapi": "3.0.0",
  "info": {
    "title": "Weather API",
    "version": "1.0.0",
    "description": "API for retrieving current weather information for cities"
  },
  "paths": {
    "/weather": {
      "get": {
        "summary": "Get current weather for a city",
        "description": "Retrieves current weather conditions including temperature, humidity, and conditions for a specified city",
        "operationId": "getWeather",
        "parameters": [
          {
            "name": "city",
            "in": "query",
            "description": "Name of the city to get weather for (e.g., Paris, London, New York)",
            "required": true,
            "schema": {
              "type": "string"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Successful response with weather data",
            "content": {
              "application/json": {
                "schema": {
                  "type": "object",
                  "properties": {
                    "city": {
                      "type": "string",
                      "description": "Name of the city"
                    },
                    "temperature_celsius": {
                      "type": "number",
                      "description": "Current temperature in Celsius"
                    },
                    "temperature_fahrenheit": {
                      "type": "number",
                      "description": "Current temperature in Fahrenheit"
                    },
                    "condition": {
                      "type": "string",
                      "description": "Current weather condition (e.g., Sunny, Rainy)"
                    },
                    "humidity": {
                      "type": "integer",
                      "description": "Current humidity percentage"
                    },
                    "wind_speed_kmh": {
                      "type": "number",
                      "description": "Wind speed in kilometers per hour"
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}
```

**Understanding the Schema:**

- **openapi**: Version of OpenAPI specification (3.0.0)
- **info**: Metadata about your API
- **paths**: Available API endpoints
  - **/weather**: The endpoint path
  - **get**: HTTP method
  - **parameters**: Input parameters (city name)
  - **responses**: Expected response structure

### 3.3 Create the Action Group in Bedrock

Back in the Bedrock Agent console:

**Step 3.3.1: Add Action Group**

1. In your agent configuration, scroll to **Action groups**
2. Click **Add**
3. Configure:
   - **Action group name:** `WeatherActionGroup`
   - **Description:** `Fetches current weather information for cities`
   - **Action group type:** `Define with API schemas`

**Step 3.3.2: Configure Action Group**

1. **Select Lambda function:**
   - Choose: `WeatherAgentFunction` (the Lambda we created)

2. **Select API Schema:**
   - **Method:** `Define with inline schema editor`
   - Paste the OpenAPI schema JSON from above

3. **Action group state:** Enable

Click **Add**

### 3.4 Grant Bedrock Permission to Invoke Lambda

This is critical! Bedrock needs permission to call your Lambda.

**Option 1: Automatic (Recommended)**
- When you add the action group, Bedrock Console may offer to add permissions automatically
- Click **Add permissions**

**Option 2: Manual (Using AWS CLI)**

```bash
aws lambda add-permission \
    --function-name WeatherAgentFunction \
    --statement-id bedrock-agent-invoke \
    --action lambda:InvokeFunction \
    --principal bedrock.amazonaws.com \
    --source-arn arn:aws:bedrock:us-east-1:YOUR_ACCOUNT_ID:agent/YOUR_AGENT_ID
```

Replace:
- `YOUR_ACCOUNT_ID` with your AWS account ID
- `YOUR_AGENT_ID` with your Bedrock agent ID

### 3.5 Prepare the Agent

After adding the action group:

1. Click **Prepare** button at the top of the agent configuration page
2. This creates a working draft of your agent with the new action group
3. Wait for the preparation to complete (usually 10-30 seconds)

**What "Prepare" Does:**
- Validates the configuration
- Creates an agent version
- Makes the agent ready for testing
- Links all components together

---

## Step 4: Invoke Agent (Console, SDK, CLI)

Now let's test our agent using three different methods!

### 4.1 Test in AWS Console (Easiest)

**Step 4.1.1: Open Test Window**

1. In the Bedrock Agent page, click the **Test** button in the top right
2. A test chat window will appear on the right side

**Step 4.1.2: Test Conversations**

Try these test queries:

**Test 1: Basic weather query**
```
User: What's the weather in Paris?
```

Expected flow:
- Agent understands the request
- Calls GetWeather action with city="Paris"
- Lambda returns mock weather data
- Agent formats a friendly response

**Test 2: Multiple cities**
```
User: Can you tell me the weather in London and Tokyo?
```

**Test 3: Conversational**
```
User: Hi there!
Agent: Hello! I'm your weather assistant...

User: I'm planning a trip to Rome
Agent: That sounds exciting! Would you like to know the current weather in Rome?

User: Yes please
Agent: [Calls GetWeather and provides response]
```

**Step 4.1.3: Examine Trace**

The console shows a "Trace" section that reveals:
- Agent's reasoning process
- Action group invocations
- Lambda function calls
- Responses at each step

This is invaluable for debugging!

### 4.2 Invoke Using AWS SDK (Python)

Create a Python script to interact with your agent programmatically.

**File: `invoke_agent.py`**

```python
import boto3
import json
from datetime import datetime

class BedrockAgentClient:
    """
    Client to interact with AWS Bedrock Agent
    """
    
    def __init__(self, agent_id, agent_alias_id, region='us-east-1'):
        """
        Initialize Bedrock Agent Runtime client
        
        Args:
            agent_id: The ID of your Bedrock agent
            agent_alias_id: The alias ID (use 'TSTALIASID' for test)
            region: AWS region where agent is deployed
        """
        self.agent_id = agent_id
        self.agent_alias_id = agent_alias_id
        self.region = region
        
        # Create Bedrock Agent Runtime client
        self.client = boto3.client(
            'bedrock-agent-runtime',
            region_name=region
        )
        
    def invoke_agent(self, prompt, session_id=None):
        """
        Invoke the Bedrock agent with a prompt
        
        Args:
            prompt: User's input text
            session_id: Optional session ID for conversation continuity
            
        Returns:
            Complete agent response as string
        """
        
        # Generate session ID if not provided
        if session_id is None:
            session_id = f"session-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        print(f"Session ID: {session_id}")
        print(f"User Input: {prompt}\n")
        
        try:
            # Invoke the agent
            response = self.client.invoke_agent(
                agentId=self.agent_id,
                agentAliasId=self.agent_alias_id,
                sessionId=session_id,
                inputText=prompt
            )
            
            # Process the streaming response
            completion = ""
            
            for event in response.get('completion', []):
                # Extract the chunk data
                if 'chunk' in event:
                    chunk = event['chunk']
                    if 'bytes' in chunk:
                        # Decode the response bytes
                        text = chunk['bytes'].decode('utf-8')
                        completion += text
                        print(text, end='', flush=True)
                
                # Handle trace events (for debugging)
                elif 'trace' in event:
                    trace = event['trace']
                    print(f"\n[TRACE]: {json.dumps(trace, indent=2)}")
            
            print("\n")
            return completion
            
        except Exception as e:
            print(f"Error invoking agent: {str(e)}")
            raise
    
    def interactive_session(self):
        """
        Start an interactive chat session with the agent
        """
        session_id = f"session-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        print("=" * 60)
        print("Bedrock Agent Interactive Session")
        print("=" * 60)
        print("Type 'exit' or 'quit' to end the session\n")
        
        while True:
            # Get user input
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['exit', 'quit']:
                print("Ending session. Goodbye!")
                break
            
            if not user_input:
                continue
            
            # Invoke agent
            print("Agent: ", end='', flush=True)
            self.invoke_agent(user_input, session_id)
            print()


def main():
    """
    Example usage of BedrockAgentClient
    """
    
    # ===== CONFIGURATION =====
    # Replace these with your actual values
    AGENT_ID = 'YOUR_AGENT_ID'  # Found in Bedrock console
    AGENT_ALIAS_ID = 'TSTALIASID'  # Use this for test alias
    REGION = 'us-east-1'  # Your AWS region
    # ========================
    
    # Create client
    agent_client = BedrockAgentClient(
        agent_id=AGENT_ID,
        agent_alias_id=AGENT_ALIAS_ID,
        region=REGION
    )
    
    # Example 1: Single query
    print("\n--- Example 1: Single Query ---")
    response = agent_client.invoke_agent("What's the weather in Paris?")
    
    # Example 2: Multiple queries in same session
    print("\n--- Example 2: Conversational ---")
    session_id = "my-session-123"
    agent_client.invoke_agent("Hi there!", session_id)
    agent_client.invoke_agent("What's the weather in Tokyo?", session_id)
    agent_client.invoke_agent("How about London?", session_id)
    
    # Example 3: Interactive mode
    print("\n--- Example 3: Interactive Mode ---")
    # Uncomment the next line to start interactive session
    # agent_client.interactive_session()


if __name__ == "__main__":
    main()
```

**How to Use:**

1. **Install boto3:**
   ```bash
   pip install boto3
   ```

2. **Configure AWS credentials:**
   ```bash
   aws configure
   ```

3. **Get your Agent ID:**
   - Go to Bedrock Console → Agents
   - Click on your agent
   - Copy the Agent ID from the top of the page

4. **Update the script:**
   - Replace `YOUR_AGENT_ID` with your actual agent ID
   - Update `REGION` if needed

5. **Run the script:**
   ```bash
   python invoke_agent.py
   ```

**Understanding the Code:**

- **invoke_agent()**: Sends a prompt to the agent and streams back the response
- **interactive_session()**: Creates a chat-like interface
- **Session ID**: Maintains conversation context across multiple invocations
- **Streaming**: Responses come back as chunks, allowing real-time display

### 4.3 Invoke Using AWS CLI

The AWS CLI provides a quick way to test your agent from the terminal.

**Step 4.3.1: Basic Invocation**

```bash
aws bedrock-agent-runtime invoke-agent \
    --agent-id YOUR_AGENT_ID \
    --agent-alias-id TSTALIASID \
    --session-id test-session-123 \
    --input-text "What's the weather in Paris?" \
    --region us-east-1 \
    response.txt
```

This saves the response to `response.txt`

**Step 4.3.2: View the Response**

```bash
cat response.txt
```

**Step 4.3.3: Formatted CLI Script**

Create a bash script for easier invocation:

**File: `invoke_agent.sh`**

```bash
#!/bin/bash

# Configuration
AGENT_ID="YOUR_AGENT_ID"
AGENT_ALIAS_ID="TSTALIASID"
REGION="us-east-1"
SESSION_ID="cli-session-$(date +%s)"

# Check if input is provided
if [ -z "$1" ]; then
    echo "Usage: ./invoke_agent.sh \"Your question here\""
    exit 1
fi

INPUT_TEXT="$1"

echo "Invoking agent with: $INPUT_TEXT"
echo "Session ID: $SESSION_ID"
echo ""

# Invoke agent and save to temp file
aws bedrock-agent-runtime invoke-agent \
    --agent-id "$AGENT_ID" \
    --agent-alias-id "$AGENT_ALIAS_ID" \
    --session-id "$SESSION_ID" \
    --input-text "$INPUT_TEXT" \
    --region "$REGION" \
    /tmp/bedrock_response.txt

# Display response
echo "Agent Response:"
echo "==============="
cat /tmp/bedrock_response.txt
echo ""
```

**Make it executable:**

```bash
chmod +x invoke_agent.sh
```

**Usage:**

```bash
./invoke_agent.sh "What's the weather in London?"
./invoke_agent.sh "Tell me about the weather in Tokyo and Paris"
```

### 4.4 Advanced: Invoke with Session State

You can pass session state to maintain context:

```python
response = client.invoke_agent(
    agentId=AGENT_ID,
    agentAliasId=AGENT_ALIAS_ID,
    sessionId=session_id,
    inputText=prompt,
    sessionState={
        'sessionAttributes': {
            'user_preference': 'celsius',
            'location': 'Europe'
        },
        'promptSessionAttributes': {
            'context': 'Planning a trip'
        }
    }
)
```

---

## Troubleshooting

### Common Issues and Solutions

#### Issue 1: Agent doesn't call the action group

**Symptoms:**
- Agent responds conversationally but doesn't fetch weather
- No Lambda invocation in traces

**Solutions:**
1. Check agent instructions mention the action explicitly
2. Ensure action group is enabled
3. Verify OpenAPI schema is valid
4. "Prepare" the agent after changes

#### Issue 2: Lambda permission errors

**Error:** `AccessDeniedException` or `Not authorized to invoke Lambda`

**Solution:**
Add Lambda permission:
```bash
aws lambda add-permission \
    --function-name WeatherAgentFunction \
    --statement-id bedrock-agent-invoke \
    --action lambda:InvokeFunction \
    --principal bedrock.amazonaws.com
```

#### Issue 3: Schema validation errors

**Symptoms:**
- Can't add action group
- Schema validation fails

**Solutions:**
1. Validate JSON syntax (use jsonlint.com)
2. Ensure all required OpenAPI fields are present
3. Check that operation IDs are unique
4. Verify response schemas match Lambda output

#### Issue 4: Agent responses are too generic

**Solutions:**
1. Improve agent instructions with specific examples
2. Add more context to OpenAPI descriptions
3. Fine-tune the foundation model selection
4. Provide clearer action group descriptions

#### Issue 5: Session not maintaining context

**Solutions:**
1. Use consistent session IDs across invocations
2. Check session timeout settings
3. Verify session state is being passed correctly
4. Don't exceed maximum session length

### Debugging Best Practices

1. **Use Console Trace**: Always check the trace in console testing
2. **CloudWatch Logs**: Enable and monitor Lambda CloudWatch logs
3. **Test Lambda Independently**: Test Lambda functions outside of Bedrock first
4. **Start Simple**: Begin with minimal instructions, then expand
5. **Version Control**: Create agent versions before major changes

### Monitoring and Logging

**Enable CloudWatch Logs for Lambda:**

```python
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def lambda_handler(event, context):
    logger.info(f"Event: {json.dumps(event)}")
    # ... rest of code
```

**Monitor Agent Invocations:**

Use CloudWatch Metrics to track:
- Invocation count
- Error rates
- Latency
- Token usage

---

## Next Steps

### Enhancements for Your Agent

1. **Add Real Weather API**
   - Integrate OpenWeatherMap or WeatherAPI
   - Handle API errors gracefully
   - Add caching to reduce API calls

2. **Multiple Action Groups**
   - Add forecast action
   - Add weather alerts action
   - Add historical weather data

3. **Knowledge Bases**
   - Add a knowledge base with weather facts
   - Include climate information
   - Add travel recommendations

4. **Guardrails**
   - Add content filtering
   - Rate limiting
   - PII detection

5. **Advanced Features**
   - Multi-language support
   - Voice integration
   - Proactive notifications

### Production Considerations

1. **Security**
   - Use least-privilege IAM roles
   - Encrypt sensitive data
   - Implement API key management

2. **Performance**
   - Optimize Lambda cold starts
   - Implement caching strategies
   - Use reserved concurrency

3. **Cost Optimization**
   - Monitor token usage
   - Set appropriate timeouts
   - Use smaller models when possible

4. **Testing**
   - Create comprehensive test cases
   - Implement CI/CD for agent updates
   - Test across different scenarios

---

## Summary

You've successfully created a Bedrock agent that:

✅ Uses foundation models (Claude) for natural conversation
✅ Has custom instructions defining its behavior
✅ Implements action groups with OpenAPI schemas
✅ Connects to Lambda functions for real functionality
✅ Can be invoked via Console, SDK, and CLI

**Key Concepts Learned:**

1. **AgentCore**: The orchestrator that uses LLMs to understand and respond
2. **Instructions**: The personality and behavioral guidelines
3. **Action Groups**: How agents perform actions via APIs
4. **OpenAPI Schemas**: The contract between agent and functionality
5. **Invocation Methods**: Multiple ways to interact with your agent

**Architecture Pattern:**
```
User Input → Agent (LLM reasoning) → Action Selection → Lambda Execution → Response
```

This foundation can be extended to build sophisticated AI applications for customer service, data analysis, workflow automation, and much more!

---

## Additional Resources

- [AWS Bedrock Documentation](https://docs.aws.amazon.com/bedrock/)
- [Bedrock Agent API Reference](https://docs.aws.amazon.com/bedrock/latest/APIReference/welcome.html)
- [OpenAPI Specification](https://swagger.io/specification/)
- [AWS Lambda Best Practices](https://docs.aws.amazon.com/lambda/latest/dg/best-practices.html)

---

**Created:** January 2026
**Use Case:** Weather Information Assistant
**Difficulty:** Beginner to Intermediate
