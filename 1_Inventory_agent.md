```markdown
# Hands-On Lab: Building Your First Amazon Bedrock Agent

## 📋 Lab Overview
In this lab, you will build **AgentCore**, an AI-powered agent using Amazon Bedrock. You will learn how to orchestrate interactions between the **Claude 3.5 Sonnet** model and a backend **Lambda function** using an **OpenAPI Schema**.

**Time to complete:** ~45 Minutes  
**Level:** Intermediate  
**Prerequisites:** AWS Account, Basic Python knowledge

---

## 🎯 Learning Objectives
By the end of this lab, you will be able to:
1.  **Create** a Bedrock Agent with specific instructions.
2.  **Deploy** a Lambda function that handles API requests from the Agent.
3.  **Define** an Action Group using an OpenAPI Schema to connect the Agent to the Lambda.
4.  **Invoke** the agent using the AWS Console and Python SDK.

---

## 🛠️ Phase 1: Deploy the Backend Logic (Lambda)

*We create the backend logic first so we have the Function ARN ready for the Agent setup.*

1.  Navigate to the **AWS Lambda Console**.
2.  Click **Create function**.
3.  **Settings:**
    * **Function name:** `InventoryAgentLogic`
    * **Runtime:** Python 3.12 (or latest)
    * **Architecture:** x86_64
4.  Click **Create function**.
5.  In the **Code Source** editor, replace the content of `lambda_function.py` with the following code.

### 🐍 Verified Lambda Code
*This code handles the event structure sent by Bedrock Agents when using OpenAPI schemas.*

```python
import json

def lambda_handler(event, context):
    print(f"Received event: {json.dumps(event)}")
    
    # 1. Parse Input
    agent = event['agent']
    actionGroup = event['actionGroup']
    apiPath = event.get('apiPath')
    httpMethod = event.get('httpMethod')
    parameters = event.get('parameters', [])
    
    response_body = {}
    
    # 2. Routing Logic
    if apiPath == '/inventory/{productId}' and httpMethod == 'GET':
        # Extract ID from parameters list safely
        product_id = next((p['value'] for p in parameters if p['name'] == 'productId'), None)
        
        if not product_id:
            response_body = {"error": "Product ID is missing"}
        elif product_id == "123":
            response_body = {
                "productId": "123", 
                "stock": 50, 
                "status": "In Stock", 
                "location": "Warehouse A"
            }
        elif product_id == "999":
            response_body = {
                "productId": "999", 
                "stock": 0, 
                "status": "Out of Stock", 
                "restockDate": "2025-01-01"
            }
        else:
            response_body = {
                "productId": product_id, 
                "stock": -1, 
                "status": "Unknown Product"
            }
    else:
        response_body = {"error": f"Path {apiPath} or method {httpMethod} not supported"}

    # 3. Format Response for Bedrock
    # CRITICAL: The body must be a JSON string nested inside 'application/json'
    action_response = {
        'messageVersion': '1.0',
        'response': {
            'actionGroup': actionGroup,
            'apiPath': apiPath,
            'httpMethod': httpMethod,
            'httpStatusCode': 200,
            'responseBody': {
                'application/json': {
                    'body': json.dumps(response_body) 
                }
            }
        }
    }
    
    return action_response

```

6. Click **Deploy**.
7. **Copy the Function ARN** (top right of the page). It looks like: `arn:aws:lambda:us-east-1:123456789:function:InventoryAgentLogic`.

---

## 🧠 Phase 2: Create the Bedrock Agent

1. Navigate to **Amazon Bedrock Console** -> **Agents**.
2. Click **Create Agent**.
3. **Agent Builder:**
* **Name:** `AgentCore`
* **Service Role:** Select **Create and use a new service role**.


4. Click **Create**.
5. **Agent Configuration:**
* **Select Model:** Anthropic **Claude 3.5 Sonnet** (recommended) or Claude 3 Sonnet.
* **Instructions:**
```text
You are AgentCore, a smart inventory assistant. 
You have access to an inventory API. 
Always check the stock level using the API when a user asks about a product. 
If the user does not provide a Product ID, ask them for it politely.

```




6. Click **Save** (disk icon at the top right).

---

## ⚡ Phase 3: Add Action Group (OpenAPI)

1. In the Agent Builder, scroll down to **Action groups** and click **Add**.
2. **Action Group Name:** `InventoryAPI`
3. **Action Group Type:** **Define with API schemas**.
4. **Action Group Invocation:**
* Select **Select an existing Lambda function**.
* Choose `InventoryAgentLogic`.


5. **API Schema:**
* Select **Define via in-line editor**.
* Paste the schema below:



```json
{
  "openapi": "3.0.0",
  "info": {
    "title": "Inventory Action Group",
    "version": "1.0.0",
    "description": "Retrieves product stock information."
  },
  "paths": {
    "/inventory/{productId}": {
      "get": {
        "summary": "Check Stock Level",
        "description": "Gets the inventory count and status for a specific product ID.",
        "operationId": "checkStock",
        "parameters": [
          {
            "name": "productId",
            "in": "path",
            "description": "The unique ID of the product (e.g., '123')",
            "required": true,
            "schema": { "type": "string" }
          }
        ],
        "responses": {
          "200": {
            "description": "Successful response",
            "content": {
              "application/json": {
                "schema": {
                  "type": "object",
                  "properties": {
                    "productId": { "type": "string" },
                    "stock": { "type": "integer" },
                    "status": { "type": "string" }
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

6. Click **Create**.

---

## 🛡️ Phase 3.5: Apply Resource Policy

*If you test the agent now, it might fail with "Access Denied" because the Lambda doesn't trust the Bedrock service. We must add a permission.*

1. Go back to your **Lambda Function** (`InventoryAgentLogic`).
2. Click the **Configuration** tab -> **Permissions** (on the left side).
3. Scroll down to **Resource-based policy statements**.
4. Click **Add permissions**.
5. Select **AWS Service**.
* **Service:** `Other` (type `bedrock` in Principal if available, otherwise use `bedrock.amazonaws.com`).
* **Statement ID:** `AllowBedrock`
* **Principal:** `bedrock.amazonaws.com`
* **Source ARN:** `arn:aws:bedrock:us-east-1:YOUR_ACCOUNT_ID:agent/*`
*(Replace `YOUR_ACCOUNT_ID` with your actual AWS Account ID).*
* **Action:** `lambda:InvokeFunction`


6. Click **Save**.

---

## 🧪 Phase 4: Invoke and Validate

### Method A: Console Test (Fastest)

1. On the right **Test Agent** panel, look for the **Prepare** banner.
2. Click **Prepare** (This compiles the DRAFT alias).
3. **Chat:** `Do we have product 123 in stock?`
4. **Expected Output:** "Yes, we have 50 units of product 123 in stock at Warehouse A."
5. **Chat:** `What about product 999?`
6. **Expected Output:** "Product 999 is currently out of stock."

### Method B: Python SDK (Production Simulation)

Create a file named `invoke_agent.py`.

```python
import boto3
import uuid

# 1. Setup Client
client = boto3.client('bedrock-agent-runtime', region_name='us-east-1')

# 2. Configuration
agent_id = 'YOUR_AGENT_ID'      # COPY FROM AGENT OVERVIEW
agent_alias_id = 'TSTALIASID'   # DEFAULT TEST ALIAS
session_id = str(uuid.uuid4())  # UNIQUE SESSION

def invoke_agent(prompt):
    print(f"\n--- Asking: {prompt} ---")
    
    response = client.invoke_agent(
        agentId=agent_id,
        agentAliasId=agent_alias_id,
        sessionId=session_id,
        inputText=prompt
    )

    # 3. Parse Stream
    for event in response.get('completion'):
        if 'chunk' in event:
            print(event['chunk']['bytes'].decode('utf-8'), end='')
    print("\n")

# Run Test
invoke_agent("Check stock for product 123")
invoke_agent("Check stock for product 999")

```

Run it:

```bash
python3 invoke_agent.py

```

### Method C: AWS CLI

```bash
aws bedrock-agent-runtime invoke-agent \
    --agent-id YOUR_AGENT_ID \
    --agent-alias-id TSTALIASID \
    --session-id cli-session-1 \
    --input-text "Check inventory for item 123"

```

---

## 🧹 Cleanup

To avoid future charges:

1. **Delete Agent:** Go to Agents -> Select `AgentCore` -> Delete.
2. **Delete Lambda:** Go to Lambda -> Select `InventoryAgentLogic` -> Delete.

```

```
