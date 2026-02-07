
# Hands-On Lab: Building Your First Amazon Bedrock Agent

## 📋 Lab Overview

In this lab, you will build **AgentCore**, an AI-powered agent using Amazon Bedrock. You will learn how to orchestrate interactions between foundation models (FMs) and software applications (Lambda functions) to execute complex tasks.

**Time to complete:** ~45 Minutes

**Level:** Intermediate

### 🎯 Learning Objectives

By the end of this lab, you will be able to:

1. Create and configure a Bedrock Agent.
2. Define instructions and select the appropriate Foundation Model.
3. **Deploy a Lambda function** to handle business logic.
4. Create Action Groups using OpenAPI schemas to connect the Agent to the Lambda.
5. Invoke the agent using the AWS Console and SDK.

---

## 🛠️ Prerequisites

* An active **AWS Account** with Administrator access.
* **Model Access:** Ensure you have access granted to **Anthropic Claude 3 Sonnet** in the Bedrock Model access settings.
* **Python 3.x** installed (for local testing).

---

## 🚀 Phase 1: Create the Backend Logic (Lambda)

*We must create the "brain" for the action first so we can select it later in the Agent setup.*

1. Navigate to the **AWS Lambda Console**.
2. Click **Create function**.
3. **Settings:**
* **Function name:** `InventoryAgentLogic`
* **Runtime:** Python 3.12 (or latest)


4. Click **Create function**.
5. Scroll down to the **Code Source** and paste the following code into `lambda_function.py`. This code handles the input from Bedrock:

```python
import json

def lambda_handler(event, context):
    print(f"Received event: {event}")
    
    # 1. Parse the Agent's request
    agent = event['agent']
    actionGroup = event['actionGroup']
    function = event['function']
    parameters = event.get('parameters', [])
    
    responseBody =  {
        "TEXT": {
            "body": "Error, no function found"
        }
    }
    
    # 2. Routing Logic
    if function == 'checkInventory':
        # Extract the productId from parameters
        product_id = next((param['value'] for param in parameters if param['name'] == 'productId'), None)
        
        # Mock Database lookup
        if product_id == "123":
            inventory_count = 50
            status = "In Stock"
        else:
            inventory_count = 0
            status = "Out of Stock"
            
        result_data = {
            "productId": product_id,
            "stockLevel": inventory_count,
            "status": status
        }
        
        # 3. Format Response for Bedrock
        responseBody = {
            "TEXT": {
                "body": json.dumps(result_data)
            }
        }

    # 4. Construct Final Return Object
    action_response = {
        'actionGroup': actionGroup,
        'function': function,
        'functionResponse': {
            'responseBody': responseBody
        }
    }
    
    return {
        'response': action_response, 
        'messageVersion': event['messageVersion']
    }

```

6. Click **Deploy**.
7. **Configuration -> Permissions:** Note the ARN of this Lambda function (top right); you will need it shortly.
8. **Resource-based policy:** Bedrock needs permission to invoke this Lambda. (The console often adds this automatically when you associate it in the next steps, but be aware of it).

---

## 🧠 Phase 2: Create the Bedrock Agent

1. Navigate to the **Amazon Bedrock Console** -> **Agents**.
2. Click **Create Agent**.
3. **Agent Details:**
* **Name:** `AgentCore`
* **Service Role:** Select "Create and use a new service role."


4. Click **Create**.
5. **Select Model:**
* Choose **Anthropic** -> **Claude 3 Sonnet**.


6. **Instructions:**
```text
You are an inventory assistant named AgentCore. 
You have access to a tool to check stock levels. 
Always check the inventory using the provided tool before answering. 
If the user asks about a product, ask for the Product ID if not provided.

```


7. Click **Save**.

---

## ⚡ Phase 3: Add Action Group

1. In the Agent Builder, scroll to **Action groups** -> **Add**.
2. **Name:** `InventoryActions`
3. **Action Group Type:** Define with API schemas.
4. **Action Group Invocation:**
* Select **Select an existing Lambda function**.
* Choose `InventoryAgentLogic` (the one created in Phase 1).


5. **API Schema:** Select **Define via in-line editor**.
6. Paste the OpenAPI Schema:

```json
{
  "openapi": "3.0.0",
  "info": {
    "title": "Inventory API",
    "version": "1.0.0",
    "description": "API to check product stock levels."
  },
  "paths": {
    "/inventory/{productId}": {
      "get": {
        "summary": "Get inventory level for a product",
        "description": "Returns the current stock count for a specific product ID.",
        "operationId": "checkInventory",
        "parameters": [
          {
            "name": "productId",
            "in": "path",
            "description": "The ID of the product to check (e.g., 123)",
            "required": true,
            "schema": {
              "type": "string"
            }
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
                    "stockLevel": { "type": "integer" },
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

7. Click **Create**.

---

## 🧪 Phase 4: Invoke and Validate

### Method A: AWS Console Test

1. On the right panel (**Test Agent**), you see a "Not Prepared" or "Prepare" banner.
2. Click **Prepare** (This builds the DRAFT alias).
3. **Chat:** "Check stock for product 123."
4. **Trace Analysis:**
* Open the **Show Trace** details.
* **Step 1 (Pre-Processing):** The agent sanitizes input.
* **Step 2 (Orchestration):** The agent identifies it needs `checkInventory` and identifies the parameter `123`.
* **Step 3 (Invocation):** It calls your Lambda.
* **Final Response:** "The stock level for product 123 is 50 and it is In Stock."



### Method B: AWS CLI

Replace `YOUR_AGENT_ID` (found in Agent overview) and ensure you use `TSTALIASID`.

```bash
aws bedrock-agent-runtime invoke-agent \
    --agent-id YOUR_AGENT_ID \
    --agent-alias-id TSTALIASID \
    --session-id test-session-01 \
    --input-text "Check stock for product 123"

```

---

## ❓ Troubleshooting

* **Error: "I'm sorry, I encountered an issue..."**
* Check CloudWatch Logs for the Lambda function. If the Lambda crashed, the error is there.
* Ensure the Lambda returns the specific JSON structure required by Bedrock (TEXT body inside functionResponse).


* **Access Denied:**
* Go to the Lambda function -> Configuration -> Permissions -> Resource-based policy statement. Ensure `bedrock.amazonaws.com` is allowed to invoke it.



---

## 🧹 Cleanup

1. Delete Agent `AgentCore`.
2. Delete Lambda `InventoryAgentLogic`.
