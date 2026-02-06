# AWS Bedrock Knowledge Base with RAG Implementation

## Hands-On Lab: Building a Production-Ready RAG Pipeline

**Estimated Duration:** 90-120 minutes  
**Difficulty Level:** Intermediate  
**Last Updated:** February 2026

## Table of Contents
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Architecture Overview](#architecture-overview)
- [Lab 1: Setup AWS Bedrock Knowledge Base](#lab-1-setup-aws-bedrock-knowledge-base)
- [Lab 2: Create and Upload Documents](#lab-2-create-and-upload-documents)
- [Lab 3: Configure Embeddings](#lab-3-configure-embeddings)
- [Lab 4: Implement RAG with LangChain](#lab-4-implement-rag-with-langchain)
- [Lab 5: Test Chunk Size Performance](#lab-5-test-chunk-size-performance)
- [Lab 6: Integrate AgentCore with RAG](#lab-6-integrate-agentcore-with-rag)
- [Troubleshooting](#troubleshooting)
- [Cleanup](#cleanup)
- [Additional Resources](#additional-resources)

---

## Overview

This hands-on lab guides you through building a complete Retrieval-Augmented Generation (RAG) system using AWS Bedrock Knowledge Base, embeddings, and LangChain. You'll learn how to:

- Create and configure an AWS Bedrock Knowledge Base
- Generate embeddings from your documents
- Implement RAG patterns using LangChain
- Optimize retrieval performance through chunk size testing
- Integrate AgentCore for advanced agent capabilities

**What is RAG?**  
Retrieval-Augmented Generation combines information retrieval with language generation. Instead of relying solely on the model's training data, RAG retrieves relevant information from your knowledge base and uses it to generate more accurate, contextual responses.

**What You'll Build:**  
A production-ready RAG pipeline that can answer questions based on your custom documents, with optimized chunk sizes and agent integration for complex workflows.

---

## Prerequisites

### Required Tools
- **AWS Account** with appropriate permissions
- **AWS CLI** (v2.x or higher) installed and configured
- **Python 3.9+** installed
- **pip** (Python package manager)
- **Git** (for version control)
- **Text editor** or IDE (VS Code recommended)

### AWS Permissions Required
Your IAM user/role needs the following permissions:
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "bedrock:*",
        "s3:*",
        "iam:CreateRole",
        "iam:AttachRolePolicy",
        "iam:PassRole",
        "opensearchserverless:*",
        "aoss:*"
      ],
      "Resource": "*"
    }
  ]
}
```

### Install Required Python Packages

**For Unix/Linux/Mac (Bash):**
```bash
# Create a virtual environment
python3 -m venv bedrock-rag-env
source bedrock-rag-env/bin/activate

# Install required packages
pip install --upgrade pip
pip install boto3 langchain langchain-aws langchain-community opensearch-py requests
```

**For Windows (PowerShell):**
```powershell
# Create a virtual environment
python -m venv bedrock-rag-env
.\bedrock-rag-env\Scripts\Activate.ps1

# Install required packages
pip install --upgrade pip
pip install boto3 langchain langchain-aws langchain-community opensearch-py requests
```

### Verify AWS CLI Configuration

```bash
# Check AWS CLI installation
aws --version

# Verify credentials
aws sts get-caller-identity

# Check Bedrock model access
aws bedrock list-foundation-models --region us-east-1
```

**Expected Output:**
```json
{
    "UserId": "AIDAXXXXXXXXXXXXXXXXX",
    "Account": "123456789012",
    "Arn": "arn:aws:iam::123456789012:user/your-username"
}
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        Your Documents                        │
│                    (PDF, TXT, MD, DOCX)                     │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
          ┌──────────────────────┐
          │     S3 Bucket        │
          │  (Document Storage)  │
          └──────────┬───────────┘
                     │
                     ▼
          ┌──────────────────────┐
          │  Bedrock Knowledge   │
          │       Base           │
          │  - Chunking          │
          │  - Embeddings        │
          └──────────┬───────────┘
                     │
                     ▼
          ┌──────────────────────┐
          │ OpenSearch Serverless│
          │  (Vector Database)   │
          └──────────┬───────────┘
                     │
                     ▼
          ┌──────────────────────┐
          │     LangChain        │
          │   RAG Pipeline       │
          └──────────┬───────────┘
                     │
                     ▼
          ┌──────────────────────┐
          │    AgentCore         │
          │  (Agent Framework)   │
          └──────────────────────┘
```

---

## Lab 1: Setup AWS Bedrock Knowledge Base

### Step 1.1: Enable Bedrock Model Access

1. Navigate to AWS Bedrock Console:
   - Go to https://console.aws.amazon.com/bedrock
   - Select your region (recommend: **us-east-1** or **us-west-2**)

2. Enable Model Access:
   - Click **"Model access"** in the left navigation
   - Click **"Enable specific models"**
   - Enable the following models:
     - ✅ **Amazon Titan Embeddings G1 - Text** (for embeddings)
     - ✅ **Anthropic Claude 3 Sonnet** (for generation)
     - ✅ **Anthropic Claude 3.5 Sonnet** (optional, better performance)
   - Click **"Save changes"**
   - Wait 2-5 minutes for activation

3. Verify Model Access:
   - The enabled models should show "Access granted" status in green

---

### Step 1.2: Create S3 Bucket for Documents

1. Navigate to S3 Console:
   - Go to https://console.aws.amazon.com/s3
   - Click **"Create bucket"**

2. Configure Bucket:
   - **Bucket name:** `bedrock-kb-docs-[your-unique-id]` (e.g., `bedrock-kb-docs-20260206`)
   - **AWS Region:** Select your region (e.g., `us-east-1`)
   - **Block Public Access:** Keep all boxes checked (recommended)
   - Leave other settings as default

3. Create Bucket:
   - Scroll to bottom and click **"Create bucket"**

4. Enable Versioning (recommended):
   - Click on your newly created bucket name
   - Go to **"Properties"** tab
   - Scroll to **"Bucket Versioning"**
   - Click **"Edit"** → Select **"Enable"** → Click **"Save changes"**

**Save your bucket name - you'll need it throughout this lab!**

---

### Step 1.3: Create IAM Role for Knowledge Base

1. Navigate to IAM Console:
   - Go to https://console.aws.amazon.com/iam
   - Click **"Roles"** in left navigation
   - Click **"Create role"**

2. Select Trusted Entity:
   - **Trusted entity type:** Select **"AWS service"**
   - **Use case:** Select **"Bedrock"**
   - Click **"Next"**

3. Add Permissions:
   - Click **"Create policy"** (opens in new tab)
   - Click **"JSON"** tab
   - Paste this policy (replace `YOUR-BUCKET-NAME` with your actual bucket name):

   ```json
   {
     "Version": "2012-10-17",
     "Statement": [
       {
         "Effect": "Allow",
         "Action": [
           "s3:GetObject",
           "s3:ListBucket"
         ],
         "Resource": [
           "arn:aws:s3:::YOUR-BUCKET-NAME",
           "arn:aws:s3:::YOUR-BUCKET-NAME/*"
         ]
       },
       {
         "Effect": "Allow",
         "Action": [
           "bedrock:InvokeModel"
         ],
         "Resource": "arn:aws:bedrock:*::foundation-model/amazon.titan-embed-text-v1"
       },
       {
         "Effect": "Allow",
         "Action": [
           "aoss:APIAccessAll"
         ],
         "Resource": "*"
       }
     ]
   }
   ```

   - Click **"Next"**
   - **Policy name:** `BedrockKnowledgeBasePolicy`
   - Click **"Create policy"**
   - Close the policy tab and return to role creation tab

4. Attach the Policy:
   - Click the refresh button (🔄) next to "Create policy"
   - Search for `BedrockKnowledgeBasePolicy`
   - Check the box next to it
   - Click **"Next"**

5. Name and Create Role:
   - **Role name:** `BedrockKnowledgeBaseRole`
   - **Description:** `Role for Bedrock Knowledge Base to access S3 and OpenSearch`
   - Click **"Create role"**

**Save the role name - you'll need it when creating the Knowledge Base!**

---

### Step 1.4: Create OpenSearch Serverless Collection

**Important:** OpenSearch Serverless is required for vector storage. Security policies must be created before the collection.

#### Step 1.4.1: Create Encryption Security Policy

The encryption policy defines how your data is encrypted at rest.

1. Navigate to OpenSearch Service Console:
   - Go to https://console.aws.amazon.com/aos
   - In the left navigation, expand **"Serverless"**
   - Click **"Security"** → **"Encryption policies"**

2. Create Encryption Policy:
   - Click **"Create encryption policy"**
   - **Policy name:** `bedrock-kb-encryption-policy`
   - **Policy definition:**
     - **Rule 1:**
       - Click **"Add rule"**
       - **Resource type:** Select `Collections`
       - **Collection name pattern:** `bedrock-kb-collection`
   - **Encryption:**
     - Select **"Use AWS managed key"**
   - Click **"Create"**

**Expected Result:** Policy shows "Active" status

---

#### Step 1.4.2: Create Network Security Policy

The network policy controls who can access your collection.

1. In OpenSearch Service Console:
   - Click **"Serverless"** → **"Security"** → **"Network policies"**

2. Create Network Policy:
   - Click **"Create network policy"**
   - **Policy name:** `bedrock-kb-network-policy`
   - **Description:** `Network access policy for Bedrock Knowledge Base` (optional)
   - **Access type:**
     - Select **"Public"** (for this lab)
     - ⚠️ *Note: In production, select "VPC" and specify your VPCs*
   - **Policy rules:**
     - Click **"Add rule"**
     - **Resource type:** Select `Collections`
     - **Collection name pattern:** `bedrock-kb-collection`
   - Click **"Create"**

**Expected Result:** Policy shows "Active" status

---

#### Step 1.4.3: Create the Collection

Now that security policies are in place, create the collection:

1. In OpenSearch Service Console:
   - Click **"Serverless"** → **"Collections"**
   - Click **"Create collection"**

2. Configure Collection Details:
   - **Collection name:** `bedrock-kb-collection`
   - **Description:** `Vector database for Bedrock Knowledge Base` (optional)
   - **Collection type:** Select **"Vector search"**

3. Configure Security:
   - **Encryption:**
     - **Select existing encryption policy:** Choose `bedrock-kb-encryption-policy`
   - **Network:**
     - **Select existing network policy:** Choose `bedrock-kb-network-policy`

4. Configure Capacity:
   - Leave at default settings (OpenSearch will auto-scale)

5. Review and Create:
   - Review all settings
   - Click **"Create"**

**Expected Result:** Collection status shows "Creating"

---

#### Step 1.4.4: Wait for Collection to Become ACTIVE

The collection takes **3-5 minutes** to provision.

1. Stay on the Collections page
2. Click the refresh button (🔄) periodically
3. Wait until **Status** changes from "Creating" to **"Active"** (shown in green)
4. Once active, click on your collection name `bedrock-kb-collection`
5. **Save the Collection Endpoint** - it looks like:
   - Example: `https://abc123xyz.us-east-1.aoss.amazonaws.com`
   - You'll need this for the vector index script!

**Visual Indicator:**
```
Status: Creating ⏳  →  Status: Active ✅
```

---

#### Step 1.4.5: Create Data Access Policy

The data access policy grants permissions to read and write data.

1. In OpenSearch Service Console:
   - Click **"Serverless"** → **"Security"** → **"Data access policies"**

2. Create Data Access Policy:
   - Click **"Create access policy"**
   - **Policy name:** `bedrock-kb-data-access-policy`
   - **Description:** `Access policy for Bedrock Knowledge Base` (optional)

3. Add Policy Rules:

   **Rule 1 - Collection Permissions:**
   - Click **"Add resource"**
   - **Resource type:** Select `Collections`
   - **Collections:** Select or enter `bedrock-kb-collection`
   - **Grant permissions:**
     - ✅ Select **ALL** collection permissions:
       - `aoss:CreateCollectionItems`
       - `aoss:DeleteCollectionItems`
       - `aoss:UpdateCollectionItems`
       - `aoss:DescribeCollectionItems`

   **Rule 2 - Index Permissions:**
   - Click **"Add another resource"**
   - **Resource type:** Select `Indexes`
   - **Index pattern:** Enter `bedrock-kb-collection/*`
   - **Grant permissions:**
     - ✅ Select **ALL** index permissions:
       - `aoss:CreateIndex`
       - `aoss:DeleteIndex`
       - `aoss:UpdateIndex`
       - `aoss:DescribeIndex`
       - `aoss:ReadDocument`
       - `aoss:WriteDocument`

4. Add Principals (Who Can Access):
   - Scroll down to **"Policy principals"** section
   - Click **"Add principals"**
   - **IAM role:** Enter `BedrockKnowledgeBaseRole`
   - The full ARN should auto-complete like: `arn:aws:iam::123456789012:role/BedrockKnowledgeBaseRole`
   - Also add your own user/role if you want to manage the collection

   **💡 How to find your Account ID:**
   - Click your username in the top-right corner
   - Your 12-digit Account ID is displayed
   - Or go to https://console.aws.amazon.com/billing/

5. Review and Create:
   - Review all settings
   - Click **"Create"**

**Expected Result:** Policy shows "Active" status

---

#### Step 1.4.6: Create Vector Index

The vector index stores document embeddings and enables similarity search. This step requires running a Python script.

**Prerequisites:**
```bash
# Install required Python packages (one-time setup)
pip install opensearch-py boto3 requests requests-aws4auth
```

**Step 1: Create the Script**

Create a new file called `create_vector_index.py` with this content:

```python
#!/usr/bin/env python3
"""
Create vector index in OpenSearch Serverless for Bedrock Knowledge Base
"""

import boto3
import sys

try:
    from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
except ImportError:
    print("❌ Missing required packages. Please run:")
    print("   pip install opensearch-py boto3 requests requests-aws4auth")
    sys.exit(1)

def create_vector_index():
    """Create vector index with proper configuration"""
    
    # Configuration - UPDATE THESE VALUES
    region = 'us-east-1'  # Change to your region
    collection_name = 'bedrock-kb-collection'
    index_name = 'bedrock-kb-index'
    vector_dimensions = 1536  # For Amazon Titan Embeddings G1
    
    print("\n" + "=" * 60)
    print("Creating Vector Index for Bedrock Knowledge Base")
    print("=" * 60)
    print(f"\n📍 Region: {region}")
    print(f"📦 Collection: {collection_name}")
    print(f"📊 Index: {index_name}")
    print(f"📐 Vector Dimensions: {vector_dimensions}")
    print("")
    
    # Connect to AWS
    try:
        session = boto3.Session(region_name=region)
        credentials = session.get_credentials()
        aoss_client = session.client('opensearchserverless')
        print("✅ Connected to AWS")
    except Exception as e:
        print(f"❌ Failed to connect to AWS: {e}")
        print("\n💡 Make sure AWS credentials are configured:")
        print("   - AWS Console: Use CloudShell")
        print("   - Local: Run 'aws configure'")
        return False
    
    # Get collection endpoint
    print("\n🔍 Finding OpenSearch collection...")
    try:
        response = aoss_client.batch_get_collection(names=[collection_name])
        
        if not response['collectionDetails']:
            print(f"❌ Collection '{collection_name}' not found!")
            print("\n💡 Make sure you:")
            print("   1. Created the collection in AWS Console")
            print("   2. Collection status is ACTIVE")
            return False
        
        collection = response['collectionDetails'][0]
        endpoint = collection['collectionEndpoint']
        status = collection['status']
        
        print(f"✅ Collection found!")
        print(f"   Endpoint: {endpoint}")
        print(f"   Status: {status}")
        
        if status != 'ACTIVE':
            print(f"\n⚠️  WARNING: Collection status is '{status}' (not ACTIVE)")
            print("   Please wait for collection to become ACTIVE first")
            return False
            
    except Exception as e:
        print(f"❌ Error accessing collection: {e}")
        return False
    
    # Connect to OpenSearch
    print("\n🔌 Connecting to OpenSearch Serverless...")
    try:
        host = endpoint.replace('https://', '')
        auth = AWSV4SignerAuth(credentials, region, 'aoss')
        
        client = OpenSearch(
            hosts=[{'host': host, 'port': 443}],
            http_auth=auth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
            timeout=300
        )
        print("✅ Connected to OpenSearch")
        
    except Exception as e:
        print(f"❌ Failed to connect to OpenSearch: {e}")
        return False
    
    # Check if index already exists
    print(f"\n🔍 Checking if index '{index_name}' exists...")
    try:
        if client.indices.exists(index=index_name):
            print(f"⚠️  Index '{index_name}' already exists!")
            print("\n💡 Index is already configured. You can proceed to create your Knowledge Base.")
            return True
    except:
        pass
    
    # Define index configuration
    print(f"\n📝 Creating index with configuration...")
    index_body = {
        "settings": {
            "index": {
                "knn": True,
                "number_of_shards": 2,
                "number_of_replicas": 0
            }
        },
        "mappings": {
            "properties": {
                "bedrock-knowledge-base-default-vector": {
                    "type": "knn_vector",
                    "dimension": vector_dimensions,
                    "method": {
                        "name": "hnsw",
                        "engine": "faiss",
                        "parameters": {
                            "ef_construction": 512,
                            "m": 16
                        },
                        "space_type": "l2"
                    }
                },
                "AMAZON_BEDROCK_METADATA": {
                    "type": "text",
                    "index": False
                },
                "AMAZON_BEDROCK_TEXT_CHUNK": {
                    "type": "text"
                }
            }
        }
    }
    
    # Create index
    try:
        response = client.indices.create(index=index_name, body=index_body)
        
        print("\n" + "=" * 60)
        print("✅ SUCCESS! Vector Index Created")
        print("=" * 60)
        print(f"\n📊 Index Details:")
        print(f"   Name: {index_name}")
        print(f"   Vector Field: bedrock-knowledge-base-default-vector")
        print(f"   Text Field: AMAZON_BEDROCK_TEXT_CHUNK")
        print(f"   Metadata Field: AMAZON_BEDROCK_METADATA")
        print(f"   Dimensions: {vector_dimensions}")
        print(f"   Algorithm: HNSW (FAISS)")
        print("")
        print("🎯 Next Step:")
        print("   Go to AWS Console and create your Bedrock Knowledge Base")
        print("   Use these EXACT field names when configuring:")
        print("   • Vector field: bedrock-knowledge-base-default-vector")
        print("   • Text field: AMAZON_BEDROCK_TEXT_CHUNK")
        print("   • Metadata field: AMAZON_BEDROCK_METADATA")
        print("")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error creating index: {e}")
        
        if "AccessDeniedException" in str(e):
            print("\n💡 PERMISSION ERROR!")
            print("   Make sure you created the data access policy (Step 1.4.5)")
            print("   The policy must include your IAM user or role")
        
        return False

def main():
    """Main execution"""
    print("\n🚀 Bedrock Knowledge Base - Vector Index Setup")
    print("   This script creates the vector index in OpenSearch Serverless")
    print("   Required for storing document embeddings\n")
    
    success = create_vector_index()
    
    if not success:
        print("\n❌ Setup failed. Please check the errors above.")
        sys.exit(1)
    
    sys.exit(0)

if __name__ == "__main__":
    main()
```

**Step 2: Run the Script**

**Option A - Using AWS CloudShell (Easiest, No Local Setup Required):**

1. In AWS Console, click the **CloudShell icon** (terminal icon `>_`) in the top navigation bar
2. Wait for CloudShell to initialize (~30 seconds)
3. Install dependencies:
   ```bash
   pip3 install opensearch-py boto3 requests requests-aws4auth --user
   ```
4. Create the script:
   ```bash
   nano create_vector_index.py
   ```
5. Copy-paste the Python script above
6. Press `Ctrl+X`, then `Y`, then `Enter` to save
7. Run the script:
   ```bash
   python3 create_vector_index.py
   ```

**Option B - Using Local Terminal:**

1. Make sure AWS CLI is configured:
   ```bash
   aws configure
   ```
2. Save the Python script as `create_vector_index.py`
3. Run:
   ```bash
   python3 create_vector_index.py
   ```

**Expected Output:**
```
============================================================
Creating Vector Index for Bedrock Knowledge Base
============================================================

📍 Region: us-east-1
📦 Collection: bedrock-kb-collection
📊 Index: bedrock-kb-index
📐 Vector Dimensions: 1536

✅ Connected to AWS

🔍 Finding OpenSearch collection...
✅ Collection found!
   Endpoint: https://abc123xyz.us-east-1.aoss.amazonaws.com
   Status: ACTIVE

🔌 Connecting to OpenSearch Serverless...
✅ Connected to OpenSearch

🔍 Checking if index 'bedrock-kb-index' exists...

📝 Creating index with configuration...

============================================================
✅ SUCCESS! Vector Index Created
============================================================

📊 Index Details:
   Name: bedrock-kb-index
   Vector Field: bedrock-knowledge-base-default-vector
   Text Field: AMAZON_BEDROCK_TEXT_CHUNK
   Metadata Field: AMAZON_BEDROCK_METADATA
   Dimensions: 1536
   Algorithm: HNSW (FAISS)

🎯 Next Step:
   Go to AWS Console and create your Bedrock Knowledge Base
   Use these EXACT field names when configuring:
   • Vector field: bedrock-knowledge-base-default-vector
   • Text field: AMAZON_BEDROCK_TEXT_CHUNK
   • Metadata field: AMAZON_BEDROCK_METADATA

============================================================
```

---

#### Step 1.4.7: Verify Setup

Before proceeding, verify everything is ready:

**Verification Checklist:**
- ✅ Encryption policy: `bedrock-kb-encryption-policy` (Status: Active)
- ✅ Network policy: `bedrock-kb-network-policy` (Status: Active)
- ✅ Collection: `bedrock-kb-collection` (Status: Active)
- ✅ Data access policy: `bedrock-kb-data-access-policy` (Status: Active)
- ✅ Vector index: `bedrock-kb-index` (Created via script)
- ✅ Collection endpoint saved

**To Verify in Console:**

1. **Check Collection:**
   - Go to: OpenSearch Service → Serverless → Collections
   - Find: `bedrock-kb-collection`
   - Status should be: **Active** (green)

2. **Check Policies:**
   - Serverless → Security → Encryption policies: `bedrock-kb-encryption-policy` ✅
   - Serverless → Security → Network policies: `bedrock-kb-network-policy` ✅
   - Serverless → Security → Data access policies: `bedrock-kb-data-access-policy` ✅

3. **Check Index:**
   - The vector index was created programmatically (not visible in console UI)
   - If the script completed successfully with "✅ SUCCESS!", your index is ready

---

### Step 1.5: Create Bedrock Knowledge Base

1. Navigate to Bedrock Console:
   - Go to https://console.aws.amazon.com/bedrock
   - Click **"Knowledge bases"** in the left menu
   - Click **"Create knowledge base"**

2. Provide Knowledge Base Details (Step 1):
   - **Name:** `my-rag-knowledge-base`
   - **Description:** `RAG knowledge base for document Q&A` (optional)
   - **IAM Role:**
     - Select **"Use an existing service role"**
     - Choose: `BedrockKnowledgeBaseRole`
   - Click **"Next"**

3. Configure Data Source (Step 2):
   - **Data source name:** `s3-documents`
   - **S3 URI:**
     - Click **"Browse S3"**
     - Select your bucket (e.g., `bedrock-kb-docs-20260206`)
     - Create a folder named `documents` if you haven't already
     - Select the `documents` folder
     - Or manually enter: `s3://your-bucket-name/documents/`
   - **Chunking and parsing settings:**
     - Leave as **"Default chunking"** (recommended)
   - Click **"Next"**

4. Select Embeddings Model and Configure Vector Store (Step 3):
   
   **Embeddings Model:**
   - **Select embeddings model:** `Titan Embeddings G1 - Text`
   - **Dimensions:** 1536 (default, matches our vector index)

   **Vector Database:**
   - **Vector database:** Select **"Amazon OpenSearch Serverless"**
   - **Select OpenSearch Serverless collection:**
     - Choose `bedrock-kb-collection` from the dropdown
   
   **Vector Store Configuration:**
   - **Vector index name:** `bedrock-kb-index`
   - **Vector field name:** `bedrock-knowledge-base-default-vector`
   - **Text field name:** `AMAZON_BEDROCK_TEXT_CHUNK`
   - **Metadata field name:** `AMAZON_BEDROCK_METADATA`

   **⚠️ CRITICAL - Field Names Must Be EXACT:**
   ```
   Vector field:   bedrock-knowledge-base-default-vector
   Text field:     AMAZON_BEDROCK_TEXT_CHUNK
   Metadata field: AMAZON_BEDROCK_METADATA
   ```
   
   **💡 Tip:** Copy-paste these field names from the output of your vector index script to avoid typos!

   - Click **"Next"**

5. Review and Create (Step 4):
   - Review all your settings
   - Verify the field names are correct
   - Click **"Create knowledge base"**
   - Wait 2-3 minutes for creation to complete

6. Save Your Knowledge Base ID:
   - Once created, you'll see your Knowledge Base details page
   - Note the **Knowledge base ID** (looks like: `ABC123XYZ`)
   - You'll need this ID for testing and integration

**Expected Result:** Knowledge Base status shows "Active" (green)

---

**✅ Checkpoint:** You should now have:
- ✅ Bedrock models enabled (Titan Embeddings, Claude)
- ✅ S3 bucket created (with `documents` folder)
- ✅ IAM role configured (BedrockKnowledgeBaseRole)
- ✅ OpenSearch security policies created (encryption, network, data access)
- ✅ OpenSearch collection active (bedrock-kb-collection)
- ✅ Vector index created (bedrock-kb-index)
- ✅ Knowledge base created with correct field mappings

---

## Lab 2: Create and Upload Documents

### Step 2.1: Prepare Sample Documents

Create a directory for your documents and add sample content:

**Bash:**
```bash
# Create documents directory
mkdir -p sample-docs
cd sample-docs

# Create sample document 1: AWS Overview
cat > aws-overview.txt << 'EOF'
Amazon Web Services (AWS) Cloud Computing Platform

AWS is a comprehensive cloud computing platform that offers over 200 fully featured services from data centers globally. Organizations use AWS to lower costs, become more agile, and innovate faster.

Key AWS Services:
- EC2 (Elastic Compute Cloud): Virtual servers in the cloud
- S3 (Simple Storage Service): Object storage with high availability
- RDS (Relational Database Service): Managed relational databases
- Lambda: Serverless computing platform
- DynamoDB: NoSQL database service

AWS operates in 32 geographic regions around the world with 102 Availability Zones. Each region is completely independent and designed to be completely isolated from other regions. This design achieves the greatest possible fault tolerance and stability.

The AWS Global Infrastructure includes:
- Regions: Physical locations around the world with multiple Availability Zones
- Availability Zones: One or more discrete data centers with redundant power and networking
- Edge Locations: Endpoints for AWS used for caching content through CloudFront
- Regional Edge Caches: Larger caches that sit between AWS services and Edge Locations

AWS follows a shared responsibility model for security:
- AWS is responsible for security OF the cloud (infrastructure, hardware, software)
- Customers are responsible for security IN the cloud (data, applications, configurations)

Pricing models include:
- Pay-as-you-go: Pay only for what you use
- Save when you reserve: Reserved instances offer significant discounts
- Pay less by using more: Volume-based discounts
- Free tier: New customers get free usage for 12 months
EOF

# Create sample document 2: Bedrock Information
cat > bedrock-intro.txt << 'EOF'
Amazon Bedrock Overview

Amazon Bedrock is a fully managed service that offers a choice of high-performing foundation models (FMs) from leading AI companies like AI21 Labs, Anthropic, Cohere, Meta, Stability AI, and Amazon via a single API. With Bedrock, you can easily experiment with and evaluate top FMs for your use case, privately customize them with your data, and build agents that execute tasks using your enterprise systems and data sources.

Key Features:
1. Choice of Leading Foundation Models
   - Anthropic Claude (multiple versions)
   - Amazon Titan models
   - AI21 Labs Jurassic models
   - Cohere Command and Embed models
   - Meta Llama models
   - Stability AI image generation models

2. Easy Customization
   - Fine-tuning with your own data
   - Continued pre-training for domain adaptation
   - No need to manage infrastructure

3. Agents for Bedrock
   - Create agents that can reason through problems
   - Break down tasks into multiple steps
   - Interact with company systems and data
   - Deliver accurate responses

4. Knowledge Bases
   - Connect FMs to your data sources
   - Implement Retrieval Augmented Generation (RAG)
   - Automatically manage embeddings and retrieval
   - Support for various document formats

5. Guardrails
   - Define content filters
   - Control model responses
   - Ensure responsible AI usage
   - Block harmful content

Bedrock Knowledge Bases allow you to build RAG applications without managing vector databases. The service handles:
- Document ingestion from S3
- Chunking and text extraction
- Embedding generation
- Vector storage in OpenSearch Serverless or other supported vector databases
- Retrieval and ranking

Use Cases:
- Conversational AI and chatbots
- Text summarization and generation
- Image generation and editing
- Search and question-answering
- Code generation and explanation
- Content personalization
EOF

# Create sample document 3: RAG Concepts
cat > rag-concepts.txt << 'EOF'
Retrieval-Augmented Generation (RAG) Concepts

What is RAG?
Retrieval-Augmented Generation is a technique that enhances Large Language Models (LLMs) by grounding their responses in retrieved, relevant information from external knowledge sources. Instead of relying solely on the model's training data, RAG dynamically retrieves pertinent information to generate more accurate and contextual responses.

RAG Architecture Components:

1. Document Processing Pipeline
   - Document ingestion: Loading documents from various sources
   - Text extraction: Converting documents to plain text
   - Chunking: Breaking text into manageable pieces
   - Embedding generation: Converting text chunks to vector representations

2. Vector Database
   - Stores document embeddings
   - Enables semantic search
   - Returns most relevant chunks based on query similarity
   - Examples: OpenSearch, Pinecone, Chroma, FAISS

3. Retrieval Mechanism
   - Query embedding: Converting user questions to vectors
   - Similarity search: Finding nearest neighbors in vector space
   - Ranking: Ordering results by relevance
   - Context selection: Choosing top-k results for generation

4. Generation Component
   - Prompt construction: Building context with retrieved information
   - LLM invocation: Generating response using context
   - Response formatting: Presenting answer to user
   - Citation tracking: Attributing sources

Chunking Strategies:

Fixed-Size Chunking:
- Split documents into chunks of fixed token/character length
- Simple and fast
- May break semantic units
- Good for uniform documents

Semantic Chunking:
- Split based on meaning and context
- Preserves logical units (paragraphs, sections)
- More complex but better quality
- Ideal for varied content

Overlapping Chunks:
- Include overlap between consecutive chunks
- Prevents context loss at boundaries
- Increases storage but improves retrieval
- Overlap typically 10-20% of chunk size

Optimal Chunk Sizes:
- Small chunks (128-256 tokens): High precision, may lack context
- Medium chunks (512-768 tokens): Balanced approach, most common
- Large chunks (1024+ tokens): More context, may dilute relevance

The choice depends on:
- Document structure and content type
- Query complexity and length
- Model context window size
- Retrieval performance requirements

RAG Evaluation Metrics:
- Retrieval Accuracy: Are the right chunks retrieved?
- Answer Relevance: Does the answer address the question?
- Faithfulness: Is the answer grounded in retrieved context?
- Context Precision: How relevant are all retrieved chunks?
- Latency: How fast is the entire pipeline?

Best Practices:
1. Clean and preprocess documents thoroughly
2. Experiment with chunk sizes for your use case
3. Use metadata to enhance retrieval
4. Implement hybrid search (keyword + semantic)
5. Monitor and log queries for continuous improvement
6. Add source attribution to generated responses
7. Handle edge cases (no results, contradictory information)
8. Implement caching for common queries
EOF

echo "Sample documents created successfully!"
ls -lh
```

**PowerShell:**
```powershell
# Create documents directory
New-Item -ItemType Directory -Force -Path "sample-docs"
Set-Location sample-docs

# Create sample document 1: AWS Overview
@"
Amazon Web Services (AWS) Cloud Computing Platform

AWS is a comprehensive cloud computing platform that offers over 200 fully featured services from data centers globally. Organizations use AWS to lower costs, become more agile, and innovate faster.

Key AWS Services:
- EC2 (Elastic Compute Cloud): Virtual servers in the cloud
- S3 (Simple Storage Service): Object storage with high availability
- RDS (Relational Database Service): Managed relational databases
- Lambda: Serverless computing platform
- DynamoDB: NoSQL database service

AWS operates in 32 geographic regions around the world with 102 Availability Zones. Each region is completely independent and designed to be completely isolated from other regions. This design achieves the greatest possible fault tolerance and stability.

The AWS Global Infrastructure includes:
- Regions: Physical locations around the world with multiple Availability Zones
- Availability Zones: One or more discrete data centers with redundant power and networking
- Edge Locations: Endpoints for AWS used for caching content through CloudFront
- Regional Edge Caches: Larger caches that sit between AWS services and Edge Locations

AWS follows a shared responsibility model for security:
- AWS is responsible for security OF the cloud (infrastructure, hardware, software)
- Customers are responsible for security IN the cloud (data, applications, configurations)

Pricing models include:
- Pay-as-you-go: Pay only for what you use
- Save when you reserve: Reserved instances offer significant discounts
- Pay less by using more: Volume-based discounts
- Free tier: New customers get free usage for 12 months
"@ | Out-File -FilePath "aws-overview.txt" -Encoding utf8

# Create sample document 2: Bedrock Information
@"
Amazon Bedrock Overview

Amazon Bedrock is a fully managed service that offers a choice of high-performing foundation models (FMs) from leading AI companies like AI21 Labs, Anthropic, Cohere, Meta, Stability AI, and Amazon via a single API. With Bedrock, you can easily experiment with and evaluate top FMs for your use case, privately customize them with your data, and build agents that execute tasks using your enterprise systems and data sources.

Key Features:
1. Choice of Leading Foundation Models
   - Anthropic Claude (multiple versions)
   - Amazon Titan models
   - AI21 Labs Jurassic models
   - Cohere Command and Embed models
   - Meta Llama models
   - Stability AI image generation models

2. Easy Customization
   - Fine-tuning with your own data
   - Continued pre-training for domain adaptation
   - No need to manage infrastructure

3. Agents for Bedrock
   - Create agents that can reason through problems
   - Break down tasks into multiple steps
   - Interact with company systems and data
   - Deliver accurate responses

4. Knowledge Bases
   - Connect FMs to your data sources
   - Implement Retrieval Augmented Generation (RAG)
   - Automatically manage embeddings and retrieval
   - Support for various document formats

5. Guardrails
   - Define content filters
   - Control model responses
   - Ensure responsible AI usage
   - Block harmful content

Bedrock Knowledge Bases allow you to build RAG applications without managing vector databases. The service handles:
- Document ingestion from S3
- Chunking and text extraction
- Embedding generation
- Vector storage in OpenSearch Serverless or other supported vector databases
- Retrieval and ranking

Use Cases:
- Conversational AI and chatbots
- Text summarization and generation
- Image generation and editing
- Search and question-answering
- Code generation and explanation
- Content personalization
"@ | Out-File -FilePath "bedrock-intro.txt" -Encoding utf8

# Create sample document 3: RAG Concepts
@"
Retrieval-Augmented Generation (RAG) Concepts

What is RAG?
Retrieval-Augmented Generation is a technique that enhances Large Language Models (LLMs) by grounding their responses in retrieved, relevant information from external knowledge sources. Instead of relying solely on the model's training data, RAG dynamically retrieves pertinent information to generate more accurate and contextual responses.

RAG Architecture Components:

1. Document Processing Pipeline
   - Document ingestion: Loading documents from various sources
   - Text extraction: Converting documents to plain text
   - Chunking: Breaking text into manageable pieces
   - Embedding generation: Converting text chunks to vector representations

2. Vector Database
   - Stores document embeddings
   - Enables semantic search
   - Returns most relevant chunks based on query similarity
   - Examples: OpenSearch, Pinecone, Chroma, FAISS

3. Retrieval Mechanism
   - Query embedding: Converting user questions to vectors
   - Similarity search: Finding nearest neighbors in vector space
   - Ranking: Ordering results by relevance
   - Context selection: Choosing top-k results for generation

4. Generation Component
   - Prompt construction: Building context with retrieved information
   - LLM invocation: Generating response using context
   - Response formatting: Presenting answer to user
   - Citation tracking: Attributing sources

Chunking Strategies:

Fixed-Size Chunking:
- Split documents into chunks of fixed token/character length
- Simple and fast
- May break semantic units
- Good for uniform documents

Semantic Chunking:
- Split based on meaning and context
- Preserves logical units (paragraphs, sections)
- More complex but better quality
- Ideal for varied content

Overlapping Chunks:
- Include overlap between consecutive chunks
- Prevents context loss at boundaries
- Increases storage but improves retrieval
- Overlap typically 10-20% of chunk size

Optimal Chunk Sizes:
- Small chunks (128-256 tokens): High precision, may lack context
- Medium chunks (512-768 tokens): Balanced approach, most common
- Large chunks (1024+ tokens): More context, may dilute relevance

The choice depends on:
- Document structure and content type
- Query complexity and length
- Model context window size
- Retrieval performance requirements

RAG Evaluation Metrics:
- Retrieval Accuracy: Are the right chunks retrieved?
- Answer Relevance: Does the answer address the question?
- Faithfulness: Is the answer grounded in retrieved context?
- Context Precision: How relevant are all retrieved chunks?
- Latency: How fast is the entire pipeline?

Best Practices:
1. Clean and preprocess documents thoroughly
2. Experiment with chunk sizes for your use case
3. Use metadata to enhance retrieval
4. Implement hybrid search (keyword + semantic)
5. Monitor and log queries for continuous improvement
6. Add source attribution to generated responses
7. Handle edge cases (no results, contradictory information)
8. Implement caching for common queries
"@ | Out-File -FilePath "rag-concepts.txt" -Encoding utf8

Write-Host "Sample documents created successfully!"
Get-ChildItem
```

---

### Step 2.2: Upload Documents to S3

**Using AWS Console:**

1. Navigate to S3 Console:
   - Go to https://console.aws.amazon.com/s3
   - Click on your bucket (e.g., `bedrock-kb-docs-20260206`)

2. Create Documents Folder (if not exists):
   - Click **"Create folder"**
   - **Folder name:** `documents`
   - Click **"Create folder"**

3. Upload Documents:
   - Click on the `documents` folder to enter it
   - Click **"Upload"**
   - Click **"Add files"**
   - Select all three `.txt` files you created:
     - `aws-overview.txt`
     - `bedrock-intro.txt`
     - `rag-concepts.txt`
   - Click **"Upload"**
   - Wait for upload to complete
   - Click **"Close"**

4. Verify Upload:
   - You should see all three files listed in the `documents` folder
   - Each file should show a size (few KB)

**Using AWS CLI (Alternative):**

```bash
# Navigate to your documents directory
cd sample-docs

# Upload all documents
aws s3 cp aws-overview.txt s3://YOUR-BUCKET-NAME/documents/
aws s3 cp bedrock-intro.txt s3://YOUR-BUCKET-NAME/documents/
aws s3 cp rag-concepts.txt s3://YOUR-BUCKET-NAME/documents/

# Verify upload
aws s3 ls s3://YOUR-BUCKET-NAME/documents/
```

---

### Step 2.3: Sync Knowledge Base with S3

After uploading documents, trigger the knowledge base to ingest and process them:

**Using AWS Console:**

1. Go to Bedrock Console:
   - Navigate to https://console.aws.amazon.com/bedrock
   - Click **"Knowledge bases"** in left menu
   - Click on your knowledge base (`my-rag-knowledge-base`)

2. Sync Data Source:
   - Click on the **"Data source"** tab
   - You should see your S3 data source (`s3-documents`)
   - Click the **checkbox** next to the data source name
   - Click **"Sync"** button at the top
   - Confirm by clicking **"Sync"** in the dialog

3. Monitor Sync Progress:
   - The "Sync status" column will show:
     - `Starting` → `In progress` → `Completed`
   - This process takes **5-10 minutes** depending on document size
   - Click the **refresh button** (🔄) to update status

4. Verify Completion:
   - Wait until "Sync status" shows **"Completed"** (green)
   - Check "Documents synced" count - should show `3`
   - Check "Last sync" timestamp - should be recent

**Expected Timeline:**
- Starting: ~30 seconds
- Processing documents: ~3-5 minutes
- Creating embeddings: ~2-4 minutes
- Total: ~5-10 minutes

**Visual Indicator:**
```
Sync Status: Starting ⏳ → In progress ⏳ → Completed ✅
Documents synced: 0 → 3
```

---

**✅ Checkpoint:** You should now have:
- ✅ Three sample documents created
- ✅ Documents uploaded to S3 (`documents` folder)
- ✅ Data source sync completed successfully
- ✅ All 3 documents processed and embedded
- ✅ Embeddings stored in vector index

---

## Lab 3: Configure Embeddings

### Understanding Embeddings

Embeddings are vector representations of text that capture semantic meaning. Similar texts have similar vectors, enabling semantic search.

**Titan Embeddings G1 - Text Specifications:**
- **Dimensions:** 1536
- **Max Input Tokens:** 8,192
- **Supported Languages:** 100+ languages
- **Best For:** English and multilingual retrieval

### Step 3.1: Test Embedding Generation

Create a Python script to test embedding generation:

**File:** `test_embeddings.py`

```python
import boto3
import json

# Initialize Bedrock client
bedrock_runtime = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-1'
)

def generate_embedding(text):
    """Generate embedding for input text."""
    body = json.dumps({
        "inputText": text
    })
    
    response = bedrock_runtime.invoke_model(
        modelId='amazon.titan-embed-text-v1',
        body=body,
        contentType='application/json',
        accept='application/json'
    )
    
    response_body = json.loads(response['body'].read())
    embedding = response_body.get('embedding')
    
    return embedding

# Test with sample texts
texts = [
    "What is AWS Lambda?",
    "How does serverless computing work?",
    "What is Amazon S3 used for?",
    "Tell me about object storage."
]

print("Generating embeddings for sample texts...\n")

embeddings = []
for i, text in enumerate(texts, 1):
    print(f"{i}. Text: {text}")
    embedding = generate_embedding(text)
    embeddings.append(embedding)
    print(f"   Embedding dimensions: {len(embedding)}")
    print(f"   First 5 values: {embedding[:5]}")
    print()

# Calculate similarity between first two embeddings
import numpy as np

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm_a = np.linalg.norm(vec1)
    norm_b = np.linalg.norm(vec2)
    return dot_product / (norm_a * norm_b)

print("\nSimilarity Analysis:")
print(f"Similarity between text 1 and 2 (related): {cosine_similarity(embeddings[0], embeddings[1]):.4f}")
print(f"Similarity between text 1 and 3 (unrelated): {cosine_similarity(embeddings[0], embeddings[2]):.4f}")
print(f"Similarity between text 3 and 4 (related): {cosine_similarity(embeddings[2], embeddings[3]):.4f}")
```

**Run the script:**
```bash
python test_embeddings.py
```

**Expected Output:**
```
Generating embeddings for sample texts...

1. Text: What is AWS Lambda?
   Embedding dimensions: 1536
   First 5 values: [0.0234, -0.0156, 0.0891, ...]

...

Similarity Analysis:
Similarity between text 1 and 2 (related): 0.8523
Similarity between text 1 and 3 (unrelated): 0.6234
Similarity between text 3 and 4 (related): 0.8912
```

---

### Step 3.2: Verify Knowledge Base Embeddings

Check that embeddings were created for your documents:

**File:** `verify_kb_embeddings.py`

```python
import boto3
import json
import os

# Configuration
KB_ID = os.environ.get('KB_ID', 'YOUR_KB_ID')  # Replace with your actual KB ID
AWS_REGION = os.environ.get('AWS_REGION', 'us-east-1')

# Initialize clients
bedrock_agent_runtime = boto3.client(
    service_name='bedrock-agent-runtime',
    region_name=AWS_REGION
)

def query_knowledge_base(query, max_results=3):
    """Query the knowledge base and return results."""
    response = bedrock_agent_runtime.retrieve(
        knowledgeBaseId=KB_ID,
        retrievalQuery={
            'text': query
        },
        retrievalConfiguration={
            'vectorSearchConfiguration': {
                'numberOfResults': max_results
            }
        }
    )
    
    return response['retrievalResults']

# Test queries
test_queries = [
    "What is AWS Lambda?",
    "Explain RAG architecture components",
    "What are the key features of Amazon Bedrock?"
]

print(f"Testing Knowledge Base: {KB_ID}\n")
print("=" * 80)

for query in test_queries:
    print(f"\nQuery: {query}")
    print("-" * 80)
    
    try:
        results = query_knowledge_base(query, max_results=2)
        
        for i, result in enumerate(results, 1):
            print(f"\nResult {i}:")
            print(f"Score: {result['score']:.4f}")
            print(f"Content: {result['content']['text'][:200]}...")
            if 'location' in result:
                print(f"Source: {result['location'].get('s3Location', {}).get('uri', 'N/A')}")
    
    except Exception as e:
        print(f"Error: {str(e)}")
    
    print("-" * 80)
```

**Run the script:**
```bash
# Set environment variable first (replace with your actual KB ID)
export KB_ID="your-actual-kb-id"
python verify_kb_embeddings.py
```

**Expected Output:**
```
Testing Knowledge Base: ABC123XYZ

================================================================================

Query: What is AWS Lambda?
--------------------------------------------------------------------------------

Result 1:
Score: 0.8234
Content: Lambda: Serverless computing platform
DynamoDB: NoSQL database service

AWS operates in 32 geographic regions around the world with 102 Availability Zones...
Source: s3://bedrock-kb-docs-20260206/documents/aws-overview.txt

Result 2:
Score: 0.7456
Content: Amazon Bedrock is a fully managed service that offers a choice of high-performing foundation models (FMs) from leading AI companies...
Source: s3://bedrock-kb-docs-20260206/documents/bedrock-intro.txt
--------------------------------------------------------------------------------
```

---

**✅ Checkpoint:** You should see:
- ✅ Documents uploaded to S3
- ✅ Ingestion job completed successfully
- ✅ Embeddings generated (1536 dimensions)
- ✅ Knowledge base returning relevant results
- ✅ Similarity scores above 0.7 for related queries

---

## Lab 4: Implement RAG with LangChain

### Step 4.1: Setup LangChain Project Structure

Create the following project structure:

```
bedrock-rag-project/
├── rag_pipeline.py          # Main RAG implementation
├── config.py                # Configuration settings
├── requirements.txt         # Python dependencies
└── test_rag.py             # Testing script
```

### Step 4.2: Create Configuration File

**File:** `config.py`

```python
import os

# AWS Configuration
AWS_REGION = os.environ.get('AWS_REGION', 'us-east-1')
KB_ID = os.environ.get('KB_ID', 'YOUR_KB_ID')

# Model Configuration
EMBEDDING_MODEL_ID = 'amazon.titan-embed-text-v1'
LLM_MODEL_ID = 'anthropic.claude-3-sonnet-20240229-v1:0'

# RAG Configuration
DEFAULT_MAX_TOKENS = 2048
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9
DEFAULT_TOP_K = 5  # Number of documents to retrieve

# Chunk Configuration (for testing)
CHUNK_SIZES = [256, 512, 768, 1024]
CHUNK_OVERLAP = 50

# OpenSearch Configuration (if using direct connection)
OPENSEARCH_ENDPOINT = os.environ.get('OPENSEARCH_ENDPOINT', '')
OPENSEARCH_INDEX = 'bedrock-kb-index'
```

---

### Step 4.3: Implement RAG Pipeline

**File:** `rag_pipeline.py`

```python
import boto3
import json
from typing import List, Dict, Optional
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_aws import BedrockLLM
from langchain_community.retrievers import AmazonKnowledgeBasesRetriever
import config

class BedrockRAGPipeline:
    """RAG Pipeline using AWS Bedrock Knowledge Base and LangChain."""
    
    def __init__(
        self,
        knowledge_base_id: str = config.KB_ID,
        region_name: str = config.AWS_REGION,
        model_id: str = config.LLM_MODEL_ID,
        max_tokens: int = config.DEFAULT_MAX_TOKENS,
        temperature: float = config.DEFAULT_TEMPERATURE,
        top_k: int = config.DEFAULT_TOP_K
    ):
        """
        Initialize the RAG pipeline.
        
        Args:
            knowledge_base_id: Bedrock Knowledge Base ID
            region_name: AWS region
            model_id: Bedrock model ID for generation
            max_tokens: Maximum tokens in response
            temperature: Model temperature (0-1)
            top_k: Number of documents to retrieve
        """
        self.knowledge_base_id = knowledge_base_id
        self.region_name = region_name
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_k = top_k
        
        # Initialize Bedrock client
        self.bedrock_runtime = boto3.client(
            service_name='bedrock-runtime',
            region_name=self.region_name
        )
        
        # Initialize LangChain components
        self._setup_retriever()
        self._setup_llm()
        self._setup_qa_chain()
    
    def _setup_retriever(self):
        """Setup the knowledge base retriever."""
        self.retriever = AmazonKnowledgeBasesRetriever(
            knowledge_base_id=self.knowledge_base_id,
            retrieval_config={
                "vectorSearchConfiguration": {
                    "numberOfResults": self.top_k
                }
            },
            region_name=self.region_name
        )
    
    def _setup_llm(self):
        """Setup the Bedrock LLM."""
        model_kwargs = {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": config.DEFAULT_TOP_P
        }
        
        self.llm = BedrockLLM(
            model_id=self.model_id,
            client=self.bedrock_runtime,
            model_kwargs=model_kwargs
        )
    
    def _setup_qa_chain(self):
        """Setup the QA chain with custom prompt."""
        prompt_template = """You are an AI assistant helping users find information from documents.
Use the following pieces of context to answer the question at the end. 
If you don't know the answer based on the context, just say that you don't know, don't try to make up an answer.
Always cite the source of your information when possible.

Context:
{context}

Question: {question}

Helpful Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
    
    def query(self, question: str) -> Dict:
        """
        Query the RAG pipeline.
        
        Args:
            question: User question
            
        Returns:
            Dictionary with answer and source documents
        """
        try:
            result = self.qa_chain.invoke({"query": question})
            
            return {
                "answer": result["result"],
                "source_documents": [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata
                    }
                    for doc in result["source_documents"]
                ],
                "success": True
            }
        except Exception as e:
            return {
                "answer": f"Error: {str(e)}",
                "source_documents": [],
                "success": False
            }
    
    def retrieve_only(self, question: str) -> List[Dict]:
        """
        Only retrieve documents without generation.
        
        Args:
            question: User question
            
        Returns:
            List of retrieved documents
        """
        try:
            docs = self.retriever.get_relevant_documents(question)
            return [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in docs
            ]
        except Exception as e:
            print(f"Retrieval error: {str(e)}")
            return []

def main():
    """Example usage of the RAG pipeline."""
    
    # Initialize pipeline
    print("Initializing RAG Pipeline...")
    rag = BedrockRAGPipeline()
    
    # Example queries
    questions = [
        "What is AWS Lambda and how does it work?",
        "Explain the key features of Amazon Bedrock.",
        "What are the components of RAG architecture?",
        "What is the difference between fixed-size and semantic chunking?"
    ]
    
    print("\nRunning example queries...\n")
    print("=" * 80)
    
    for i, question in enumerate(questions, 1):
        print(f"\nQuestion {i}: {question}")
        print("-" * 80)
        
        # Query the pipeline
        result = rag.query(question)
        
        if result["success"]:
            print(f"\nAnswer:\n{result['answer']}\n")
            print(f"Sources ({len(result['source_documents'])} documents):")
            for j, doc in enumerate(result['source_documents'], 1):
                print(f"\n  Source {j}:")
                print(f"  {doc['content'][:150]}...")
        else:
            print(f"\nError: {result['answer']}")
        
        print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
```

---

### Step 4.4: Create Test Script

**File:** `test_rag.py`

```python
#!/usr/bin/env python3
"""
Test script for RAG pipeline functionality.
"""

import sys
from rag_pipeline import BedrockRAGPipeline
import config

def test_retrieval_only():
    """Test document retrieval without generation."""
    print("\n" + "=" * 80)
    print("TEST 1: Document Retrieval Only")
    print("=" * 80)
    
    rag = BedrockRAGPipeline()
    question = "What is Amazon Bedrock?"
    
    print(f"\nQuery: {question}\n")
    docs = rag.retrieve_only(question)
    
    print(f"Retrieved {len(docs)} documents:\n")
    for i, doc in enumerate(docs, 1):
        print(f"Document {i}:")
        print(f"Content: {doc['content'][:200]}...")
        print(f"Metadata: {doc['metadata']}")
        print()

def test_full_rag():
    """Test full RAG pipeline with generation."""
    print("\n" + "=" * 80)
    print("TEST 2: Full RAG Pipeline (Retrieval + Generation)")
    print("=" * 80)
    
    rag = BedrockRAGPipeline()
    question = "What are the main components of a RAG architecture?"
    
    print(f"\nQuery: {question}\n")
    result = rag.query(question)
    
    if result["success"]:
        print("Answer:")
        print(result["answer"])
        print(f"\nUsed {len(result['source_documents'])} source documents")
    else:
        print(f"Error: {result['answer']}")

def test_multiple_queries():
    """Test multiple queries to evaluate consistency."""
    print("\n" + "=" * 80)
    print("TEST 3: Multiple Query Test")
    print("=" * 80)
    
    rag = BedrockRAGPipeline()
    
    queries = [
        "What is serverless computing?",
        "How does AWS pricing work?",
        "What are the benefits of using Knowledge Bases in Bedrock?"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\nQuery {i}: {query}")
        print("-" * 80)
        result = rag.query(query)
        if result["success"]:
            print(f"Answer: {result['answer'][:300]}...")
        else:
            print(f"Error: {result['answer']}")

def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n" + "=" * 80)
    print("TEST 4: Edge Cases")
    print("=" * 80)
    
    rag = BedrockRAGPipeline()
    
    edge_cases = [
        "",  # Empty query
        "x" * 1000,  # Very long query
        "What is the meaning of life?",  # Out of domain
    ]
    
    for i, query in enumerate(edge_cases, 1):
        print(f"\nEdge Case {i}: {query[:50]}{'...' if len(query) > 50 else ''}")
        result = rag.query(query)
        print(f"Success: {result['success']}")
        if result["success"]:
            print(f"Answer length: {len(result['answer'])} characters")

def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("RAG PIPELINE TEST SUITE")
    print("=" * 80)
    print(f"\nKnowledge Base ID: {config.KB_ID}")
    print(f"Region: {config.AWS_REGION}")
    print(f"Model: {config.LLM_MODEL_ID}")
    
    try:
        test_retrieval_only()
        test_full_rag()
        test_multiple_queries()
        test_edge_cases()
        
        print("\n" + "=" * 80)
        print("ALL TESTS COMPLETED")
        print("=" * 80)
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

---

### Step 4.5: Run the RAG Pipeline

**Set Environment Variables:**

```bash
# Set your Knowledge Base ID
export KB_ID="your-knowledge-base-id"
export AWS_REGION="us-east-1"
```

**Run the main pipeline:**
```bash
python rag_pipeline.py
```

**Run tests:**
```bash
python test_rag.py
```

**Expected Output:**
```
Initializing RAG Pipeline...

Running example queries...

================================================================================

Question 1: What is AWS Lambda and how does it work?
--------------------------------------------------------------------------------

Answer:
AWS Lambda is a serverless computing platform offered by Amazon Web Services. It allows you to run code without provisioning or managing servers...

Sources (3 documents):

  Source 1:
  Lambda: Serverless computing platform
DynamoDB: NoSQL database service...

================================================================================
```

---

**✅ Checkpoint:** You should see:
- ✅ Documents successfully retrieved
- ✅ LLM generating contextual answers
- ✅ Source citations included
- ✅ All tests passing

---

## Lab 5: Test Chunk Size Performance

Understanding how chunk size affects retrieval quality is critical for optimizing your RAG system.

### Step 5.1: Create Chunk Size Testing Script

**File:** `test_chunk_sizes.py`

```python
#!/usr/bin/env python3
"""
Test different chunk sizes and compare retrieval performance.
"""

import boto3
import json
import time
from typing import List, Dict, Tuple
from datetime import datetime
import config

class ChunkSizeTester:
    """Test retrieval performance with different chunk configurations."""
    
    def __init__(self, kb_id: str, region: str = 'us-east-1'):
        self.kb_id = kb_id
        self.region = region
        self.bedrock_agent_runtime = boto3.client(
            service_name='bedrock-agent-runtime',
            region_name=region
        )
    
    def retrieve_with_config(
        self,
        query: str,
        num_results: int = 5
    ) -> Tuple[List[Dict], float]:
        """
        Retrieve documents and measure latency.
        
        Returns:
            Tuple of (results, latency_ms)
        """
        start_time = time.time()
        
        try:
            response = self.bedrock_agent_runtime.retrieve(
                knowledgeBaseId=self.kb_id,
                retrievalQuery={'text': query},
                retrievalConfiguration={
                    'vectorSearchConfiguration': {
                        'numberOfResults': num_results
                    }
                }
            )
            
            latency = (time.time() - start_time) * 1000  # Convert to ms
            results = response['retrievalResults']
            
            return results, latency
            
        except Exception as e:
            print(f"Error during retrieval: {str(e)}")
            return [], 0.0
    
    def evaluate_results(
        self,
        results: List[Dict],
        query: str
    ) -> Dict:
        """
        Evaluate quality of retrieved results.
        
        Returns:
            Dictionary with evaluation metrics
        """
        if not results:
            return {
                'avg_score': 0.0,
                'max_score': 0.0,
                'min_score': 0.0,
                'avg_length': 0,
                'num_results': 0
            }
        
        scores = [r['score'] for r in results]
        lengths = [len(r['content']['text']) for r in results]
        
        return {
            'avg_score': sum(scores) / len(scores),
            'max_score': max(scores),
            'min_score': min(scores),
            'avg_length': sum(lengths) // len(lengths),
            'num_results': len(results)
        }
    
    def run_test_suite(self, test_queries: List[str]) -> Dict:
        """
        Run comprehensive test suite with multiple queries.
        
        Args:
            test_queries: List of test questions
            
        Returns:
            Dictionary with aggregated results
        """
        print("\n" + "=" * 80)
        print("CHUNK SIZE PERFORMANCE TEST")
        print("=" * 80)
        print(f"Knowledge Base: {self.kb_id}")
        print(f"Test Queries: {len(test_queries)}")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        all_results = []
        
        for i, query in enumerate(test_queries, 1):
            print(f"\nQuery {i}/{len(test_queries)}: {query}")
            print("-" * 80)
            
            results, latency = self.retrieve_with_config(query, num_results=5)
            metrics = self.evaluate_results(results, query)
            
            print(f"Latency: {latency:.2f} ms")
            print(f"Results Retrieved: {metrics['num_results']}")
            print(f"Average Relevance Score: {metrics['avg_score']:.4f}")
            print(f"Average Chunk Length: {metrics['avg_length']} characters")
            
            # Display top result
            if results:
                print(f"\nTop Result (Score: {results[0]['score']:.4f}):")
                print(f"{results[0]['content']['text'][:200]}...")
            
            all_results.append({
                'query': query,
                'latency_ms': latency,
                'metrics': metrics,
                'results': results
            })
        
        # Calculate aggregate statistics
        avg_latency = sum(r['latency_ms'] for r in all_results) / len(all_results)
        avg_score = sum(r['metrics']['avg_score'] for r in all_results) / len(all_results)
        avg_length = sum(r['metrics']['avg_length'] for r in all_results) / len(all_results)
        
        print("\n" + "=" * 80)
        print("AGGREGATE STATISTICS")
        print("=" * 80)
        print(f"Average Latency: {avg_latency:.2f} ms")
        print(f"Average Relevance Score: {avg_score:.4f}")
        print(f"Average Chunk Length: {avg_length} characters")
        
        return {
            'individual_results': all_results,
            'aggregate': {
                'avg_latency_ms': avg_latency,
                'avg_relevance_score': avg_score,
                'avg_chunk_length': avg_length
            }
        }

def compare_chunk_strategies():
    """
    Compare different chunking strategies.
    
    Note: Since AWS Bedrock Knowledge Base handles chunking automatically,
    this function demonstrates how you would test if you had control over
    chunk size configurations.
    """
    print("\n" + "=" * 80)
    print("CHUNKING STRATEGY COMPARISON")
    print("=" * 80)
    print("""
Current Knowledge Base Configuration:
    
When you created your Knowledge Base, AWS Bedrock automatically configured
the chunking strategy. The default settings are:

- Chunk Size: ~300 tokens (~400-500 characters)
- Overlap: ~20% between chunks
- Strategy: Semantic chunking with sentence boundary detection

To test different chunk sizes, you would need to:
1. Create multiple Knowledge Bases with different configurations
2. OR manually process documents and upload pre-chunked content
3. OR use custom data sources with different preprocessing

For this lab, we're testing the default configuration.
If you want to experiment with different chunk sizes, you can:
- Re-create your KB with different chunking parameters in the console
- Use the Knowledge Base API with custom chunking configurations
    """)

def main():
    """Main execution function."""
    
    # Test queries covering different topics and complexities
    test_queries = [
        # Short, specific queries
        "What is AWS Lambda?",
        "What is Amazon Bedrock?",
        
        # Medium complexity queries
        "Explain the components of RAG architecture",
        "How does AWS pricing work?",
        
        # Complex, multi-part queries
        "What are the differences between fixed-size chunking and semantic chunking, and when should I use each?",
        "Describe the complete workflow of creating a Knowledge Base in Amazon Bedrock",
        
        # Domain-specific queries
        "What are the best practices for RAG implementation?",
        "How do I optimize retrieval performance in a knowledge base?"
    ]
    
    # Initialize tester
    tester = ChunkSizeTester(
        kb_id=config.KB_ID,
        region=config.AWS_REGION
    )
    
    # Run tests
    results = tester.run_test_suite(test_queries)
    
    # Save results to file
    output_file = f"chunk_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_file}")
    
    # Show strategy comparison info
    compare_chunk_strategies()

if __name__ == "__main__":
    main()
```

---

### Step 5.2: Analyze Chunk Size Results

**File:** `analyze_chunks.py`

```python
#!/usr/bin/env python3
"""
Analyze chunk size test results and provide recommendations.
"""

import json
import glob
from typing import Dict, List

def analyze_results(results: Dict) -> None:
    """Analyze test results and provide insights."""
    
    print("\n" + "=" * 80)
    print("CHUNK SIZE ANALYSIS")
    print("=" * 80)
    
    aggregate = results['aggregate']
    individual = results['individual_results']
    
    # Overall performance
    print("\nOverall Performance:")
    print(f"  Average Latency: {aggregate['avg_latency_ms']:.2f} ms")
    print(f"  Average Relevance: {aggregate['avg_relevance_score']:.4f}")
    print(f"  Average Chunk Length: {aggregate['avg_chunk_length']} chars")
    
    # Find best and worst performing queries
    best_query = max(individual, key=lambda x: x['metrics']['avg_score'])
    worst_query = min(individual, key=lambda x: x['metrics']['avg_score'])
    
    print("\nBest Performing Query:")
    print(f"  Query: {best_query['query']}")
    print(f"  Avg Score: {best_query['metrics']['avg_score']:.4f}")
    print(f"  Latency: {best_query['latency_ms']:.2f} ms")
    
    print("\nWorst Performing Query:")
    print(f"  Query: {worst_query['query']}")
    print(f"  Avg Score: {worst_query['metrics']['avg_score']:.4f}")
    print(f"  Latency: {worst_query['latency_ms']:.2f} ms")
    
    # Recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    
    if aggregate['avg_relevance_score'] > 0.7:
        print("\n✅ Good: Relevance scores are high")
        print("   Current chunk size appears optimal for your documents")
    elif aggregate['avg_relevance_score'] > 0.5:
        print("\n⚠️  Moderate: Relevance scores could be improved")
        print("   Consider:")
        print("   - Adjusting chunk overlap")
        print("   - Using semantic chunking")
        print("   - Adding more diverse documents")
    else:
        print("\n❌ Low: Relevance scores need improvement")
        print("   Actions needed:")
        print("   - Review document quality and relevance")
        print("   - Consider smaller chunks for better precision")
        print("   - Ensure documents match your query domain")
    
    if aggregate['avg_latency_ms'] < 500:
        print("\n✅ Good: Latency is acceptable (<500ms)")
    elif aggregate['avg_latency_ms'] < 1000:
        print("\n⚠️  Moderate: Latency is noticeable (500-1000ms)")
        print("   Consider caching frequent queries")
    else:
        print("\n❌ High: Latency is too slow (>1000ms)")
        print("   Optimization needed:")
        print("   - Reduce number of retrieved documents")
        print("   - Check network connectivity")
        print("   - Consider caching layer")
    
    if aggregate['avg_chunk_length'] < 300:
        print("\n📊 Chunk Size: Small chunks (more precise, less context)")
    elif aggregate['avg_chunk_length'] < 700:
        print("\n📊 Chunk Size: Medium chunks (balanced approach)")
    else:
        print("\n📊 Chunk Size: Large chunks (more context, less precise)")

def main():
    """Load and analyze most recent test results."""
    
    # Find most recent results file
    result_files = glob.glob("chunk_test_results_*.json")
    
    if not result_files:
        print("No test results found. Run test_chunk_sizes.py first.")
        return
    
    latest_file = max(result_files)
    print(f"Analyzing: {latest_file}")
    
    with open(latest_file, 'r') as f:
        results = json.load(f)
    
    analyze_results(results)

if __name__ == "__main__":
    main()
```

---

### Step 5.3: Run Chunk Size Tests

```bash
# Run chunk size tests
python test_chunk_sizes.py

# Analyze results
python analyze_chunks.py
```

**Expected Output:**
```
================================================================================
CHUNK SIZE PERFORMANCE TEST
================================================================================
Knowledge Base: ABC123XYZ
Test Queries: 8
Timestamp: 2026-02-06 10:30:45
================================================================================

Query 1/8: What is AWS Lambda?
--------------------------------------------------------------------------------
Latency: 245.67 ms
Results Retrieved: 5
Average Relevance Score: 0.8123
Average Chunk Length: 456 characters

Top Result (Score: 0.8456):
Lambda: Serverless computing platform...

...

================================================================================
AGGREGATE STATISTICS
================================================================================
Average Latency: 287.34 ms
Average Relevance Score: 0.7845
Average Chunk Length: 423 characters

Results saved to: chunk_test_results_20260206_103045.json
```

---

**✅ Checkpoint:** You should see:
- ✅ Performance metrics for each query
- ✅ Latency measurements (< 500ms ideal)
- ✅ Relevance score analysis (> 0.7 is good)
- ✅ Recommendations for optimization

---

## Lab 6: Integrate AgentCore with RAG

### Step 6.1: Understanding AgentCore Integration

AgentCore allows you to create autonomous agents that can:
- Use tools and APIs
- Make decisions based on context
- Execute multi-step workflows
- Integrate with your RAG pipeline

### Step 6.2: Create Agent with RAG Integration

**File:** `agent_with_rag.py`

```python
#!/usr/bin/env python3
"""
AgentCore integration with RAG pipeline for complex workflows.
"""

import boto3
import json
from typing import List, Dict, Optional
from datetime import datetime
import config
from rag_pipeline import BedrockRAGPipeline

class RAGAgent:
    """Agent that combines RAG retrieval with reasoning and tool use."""
    
    def __init__(
        self,
        knowledge_base_id: str = config.KB_ID,
        region_name: str = config.AWS_REGION
    ):
        """Initialize the RAG Agent."""
        self.region_name = region_name
        self.knowledge_base_id = knowledge_base_id
        
        # Initialize clients
        self.bedrock_runtime = boto3.client(
            service_name='bedrock-runtime',
            region_name=region_name
        )
        
        self.bedrock_agent = boto3.client(
            service_name='bedrock-agent',
            region_name=region_name
        )
        
        self.bedrock_agent_runtime = boto3.client(
            service_name='bedrock-agent-runtime',
            region_name=region_name
        )
        
        # Initialize RAG pipeline
        self.rag_pipeline = BedrockRAGPipeline(
            knowledge_base_id=knowledge_base_id,
            region_name=region_name
        )
        
        # Conversation history
        self.conversation_history = []
    
    def process_with_reasoning(
        self,
        query: str,
        use_rag: bool = True
    ) -> Dict:
        """
        Process query with agent reasoning and optional RAG.
        
        Args:
            query: User query
            use_rag: Whether to use RAG retrieval
            
        Returns:
            Dictionary with response and reasoning chain
        """
        reasoning_steps = []
        
        # Step 1: Analyze query
        reasoning_steps.append({
            'step': 'analyze_query',
            'description': 'Analyzing user query to determine intent',
            'query': query
        })
        
        # Step 2: Decide if RAG is needed
        needs_rag = self._needs_knowledge_base(query)
        reasoning_steps.append({
            'step': 'determine_rag_need',
            'description': 'Determining if knowledge base retrieval is needed',
            'needs_rag': needs_rag and use_rag
        })
        
        # Step 3: Retrieve from knowledge base if needed
        context = None
        if needs_rag and use_rag:
            rag_result = self.rag_pipeline.query(query)
            context = rag_result.get('answer', '')
            
            reasoning_steps.append({
                'step': 'rag_retrieval',
                'description': 'Retrieved relevant documents from knowledge base',
                'num_sources': len(rag_result.get('source_documents', [])),
                'context_length': len(context)
            })
        
        # Step 4: Generate response
        response = self._generate_response(query, context)
        reasoning_steps.append({
            'step': 'generate_response',
            'description': 'Generated final response',
            'response_length': len(response)
        })
        
        # Step 5: Add to conversation history
        self.conversation_history.append({
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'response': response,
            'used_rag': needs_rag and use_rag
        })
        
        return {
            'response': response,
            'reasoning_steps': reasoning_steps,
            'used_rag': needs_rag and use_rag,
            'context': context
        }
    
    def _needs_knowledge_base(self, query: str) -> bool:
        """
        Determine if query requires knowledge base retrieval.
        
        Simple heuristic - in production, use more sophisticated classification.
        """
        # Keywords that suggest factual information needs
        knowledge_keywords = [
            'what is', 'what are', 'how does', 'explain',
            'describe', 'tell me about', 'definition',
            'features', 'components', 'architecture'
        ]
        
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in knowledge_keywords)
    
    def _generate_response(
        self,
        query: str,
        context: Optional[str] = None
    ) -> str:
        """Generate response using Bedrock model."""
        
        if context:
            prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {query}

Answer:"""
        else:
            prompt = query
        
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 2048,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        })
        
        try:
            response = self.bedrock_runtime.invoke_model(
                modelId=config.LLM_MODEL_ID,
                body=body
            )
            
            response_body = json.loads(response['body'].read())
            return response_body['content'][0]['text']
            
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def multi_step_workflow(self, task: str) -> Dict:
        """
        Execute multi-step workflow combining RAG and reasoning.
        
        Example: "Summarize AWS services and compare their pricing"
        """
        print("\n" + "=" * 80)
        print(f"MULTI-STEP WORKFLOW: {task}")
        print("=" * 80)
        
        workflow_steps = []
        
        # Step 1: Break down task
        print("\n🔍 Step 1: Breaking down task...")
        subtasks = self._decompose_task(task)
        workflow_steps.append({
            'step': 'task_decomposition',
            'subtasks': subtasks
        })
        
        # Step 2: Execute each subtask
        results = []
        for i, subtask in enumerate(subtasks, 1):
            print(f"\n⚙️  Step {i+1}: Executing subtask - {subtask}")
            result = self.process_with_reasoning(subtask)
            results.append(result)
            workflow_steps.append({
                'step': f'subtask_{i}',
                'query': subtask,
                'result': result
            })
        
        # Step 3: Synthesize results
        print("\n🔄 Final Step: Synthesizing results...")
        final_answer = self._synthesize_results(task, results)
        workflow_steps.append({
            'step': 'synthesis',
            'final_answer': final_answer
        })
        
        print("\n✅ Workflow completed!")
        
        return {
            'task': task,
            'workflow_steps': workflow_steps,
            'final_answer': final_answer
        }
    
    def _decompose_task(self, task: str) -> List[str]:
        """Break down complex task into subtasks."""
        # Simple decomposition - in production, use LLM for this
        if "summarize" in task.lower() and "compare" in task.lower():
            return [
                "What are the main AWS services?",
                "How does AWS pricing work?",
                "Compare pricing models across services"
            ]
        elif "explain" in task.lower():
            return [
                f"What is {task.replace('explain', '').strip()}?",
                f"What are the key components?",
                f"What are the use cases?"
            ]
        else:
            return [task]
    
    def _synthesize_results(
        self,
        original_task: str,
        results: List[Dict]
    ) -> str:
        """Synthesize multiple results into final answer."""
        
        # Combine all responses
        combined_context = "\n\n".join([
            f"Subtask: {r.get('reasoning_steps', [{}])[0].get('query', '')}\n"
            f"Answer: {r['response']}"
            for r in results
        ])
        
        synthesis_prompt = f"""Based on the following information, provide a comprehensive answer to this task:

Task: {original_task}

Information:
{combined_context}

Comprehensive Answer:"""
        
        return self._generate_response(original_task, synthesis_prompt)

def main():
    """Example usage of RAG Agent."""
    
    print("\n" + "=" * 80)
    print("RAG AGENT WITH AGENTCORE INTEGRATION")
    print("=" * 80)
    
    # Initialize agent
    agent = RAGAgent()
    
    # Example 1: Simple query with reasoning
    print("\n\nExample 1: Simple Query with Reasoning")
    print("=" * 80)
    
    result = agent.process_with_reasoning(
        "What is Amazon Bedrock?"
    )
    
    print(f"\nResponse: {result['response']}")
    print(f"\nUsed RAG: {result['used_rag']}")
    print("\nReasoning Steps:")
    for step in result['reasoning_steps']:
        print(f"  - {step['description']}")
    
    # Example 2: Multi-step workflow
    print("\n\nExample 2: Multi-Step Workflow")
    print("=" * 80)
    
    workflow_result = agent.multi_step_workflow(
        "Explain RAG architecture and its main components"
    )
    
    print(f"\n\nFinal Answer:")
    print(workflow_result['final_answer'])
    
    # Example 3: Conversation with history
    print("\n\nExample 3: Conversation History")
    print("=" * 80)
    
    queries = [
        "What is AWS Lambda?",
        "How does it differ from EC2?",
        "What are the pricing models?"
    ]
    
    for query in queries:
        result = agent.process_with_reasoning(query)
        print(f"\nQ: {query}")
        print(f"A: {result['response'][:200]}...")
    
    print(f"\n\nConversation history: {len(agent.conversation_history)} exchanges")

if __name__ == "__main__":
    main()
```

---

### Step 6.3: Test Agent Integration

```bash
# Set environment variables
export KB_ID="your-knowledge-base-id"
export AWS_REGION="us-east-1"

# Run the agent
python agent_with_rag.py
```

**Expected Output:**
```
================================================================================
RAG AGENT WITH AGENTCORE INTEGRATION
================================================================================


Example 1: Simple Query with Reasoning
================================================================================

Response: Amazon Bedrock is a fully managed service that offers access to high-performing foundation models from leading AI companies...

Used RAG: True

Reasoning Steps:
  - Analyzing user query to determine intent
  - Determining if knowledge base retrieval is needed
  - Retrieved relevant documents from knowledge base
  - Generated final response


Example 2: Multi-Step Workflow
================================================================================

🔍 Step 1: Breaking down task...

⚙️  Step 2: Executing subtask - What is RAG architecture and its main components?

⚙️  Step 3: Executing subtask - What are the key components?

⚙️  Step 4: Executing subtask - What are the use cases?

🔄 Final Step: Synthesizing results...

✅ Workflow completed!


Final Answer:
RAG (Retrieval-Augmented Generation) architecture consists of four main components: the Document Processing Pipeline for ingestion and chunking, a Vector Database for storing embeddings, a Retrieval Mechanism for semantic search, and a Generation Component that uses retrieved context to produce responses...
```

---

**✅ Checkpoint:** You should see:
- ✅ Agent processing queries with reasoning
- ✅ RAG integration working
- ✅ Multi-step workflows executing
- ✅ Conversation history tracking

---

## Troubleshooting

### Common Issues and Solutions

#### Issue 1: Knowledge Base Not Returning Results

**Symptoms:**
- Empty results from queries
- Low relevance scores
- "No documents found" errors

**Solutions:**

1. **Check Data Source Sync:**
   - Go to: Bedrock Console → Knowledge bases → Your KB → Data source tab
   - Verify "Sync status" is **"Completed"**
   - Check "Documents synced" count (should be > 0)
   - If needed, click **"Sync"** to re-sync

2. **Verify Documents in S3:**
   - Go to: S3 Console → Your bucket → `documents` folder
   - Confirm files are present and not empty
   - Check file permissions

3. **Re-sync if Necessary:**
   - Select your data source
   - Click **"Sync"**
   - Wait 5-10 minutes for completion

---

#### Issue 2: "Access Denied" Errors

**Symptoms:**
- 403 Forbidden errors
- "Not authorized to perform action"

**Solutions:**

1. **Check IAM Role Permissions:**
   - Go to: IAM Console → Roles → BedrockKnowledgeBaseRole
   - Verify policy includes:
     - S3 read access to your bucket
     - Bedrock InvokeModel permission
     - OpenSearch API access

2. **Verify Model Access:**
   - Go to: Bedrock Console → Model access
   - Confirm "Access granted" for Titan Embeddings and Claude

3. **Check Data Access Policy:**
   - Go to: OpenSearch Console → Serverless → Security → Data access policies
   - Verify `bedrock-kb-data-access-policy` includes your role

---

#### Issue 3: Python Package Import Errors

**Symptoms:**
```
ModuleNotFoundError: No module named 'langchain_aws'
```

**Solutions:**

```bash
# Reinstall packages
pip install --upgrade pip
pip install --upgrade langchain langchain-aws langchain-community opensearch-py

# Verify installation
pip list | grep langchain

# If using virtual environment, ensure it's activated
source bedrock-rag-env/bin/activate  # Unix/Mac
.\bedrock-rag-env\Scripts\Activate.ps1  # Windows
```

---

#### Issue 4: Vector Field Name Mismatch

**Symptoms:**
```
Error: Field 'bedrock-kb-vector' is not knn_vector type
```

**Solution:**

The field name must be **EXACTLY**: `bedrock-knowledge-base-default-vector`

1. **In AWS Console when creating Knowledge Base:**
   - Step 3: Configure data storage
   - **Vector field name:** `bedrock-knowledge-base-default-vector` (copy-paste this!)
   - **Text field name:** `AMAZON_BEDROCK_TEXT_CHUNK`
   - **Metadata field name:** `AMAZON_BEDROCK_METADATA`

2. **Verify Field Names:**
   - These must match what was created in the vector index script
   - Copy-paste from the script output to avoid typos

---

#### Issue 5: Slow Retrieval Performance

**Symptoms:**
- Queries taking >2 seconds
- Timeouts

**Solutions:**

1. **Reduce Retrieved Documents:**
   ```python
   # In your code
   retrieval_config={
       "vectorSearchConfiguration": {
           "numberOfResults": 3  # Reduced from 5
       }
   }
   ```

2. **Check OpenSearch Collection Health:**
   - Go to: OpenSearch Console → Serverless → Collections
   - Verify `bedrock-kb-collection` status is **Active**

3. **Consider Caching:**
   - Implement caching for frequent queries
   - Use in-memory cache (Redis) for production

---

#### Issue 6: Collection Creation Failed

**Symptoms:**
- Collection stuck in "Creating" state
- Error during collection creation

**Solutions:**

1. **Verify Security Policies Exist:**
   - Encryption policy: `bedrock-kb-encryption-policy`
   - Network policy: `bedrock-kb-network-policy`
   - These MUST exist before creating collection

2. **Check Policy Configuration:**
   - Policy rules must match collection name pattern
   - Resource type must be correct

3. **Recreate if Necessary:**
   - Delete failed collection
   - Verify policies exist
   - Recreate collection

---

### Debug Mode

Enable detailed logging in Python:

```python
import logging

# Enable debug logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

---

### Getting Help

If issues persist:

1. **AWS Service Health Dashboard:**
   - Check: https://health.aws.amazon.com/health/status

2. **CloudWatch Logs:**
   - Go to: CloudWatch → Log groups
   - Look for Bedrock and OpenSearch logs

3. **AWS Support:**
   - File a support case if you have a support plan
   - Include error messages and timestamps

4. **AWS re:Post Community:**
   - Ask questions: https://repost.aws/

---

## Cleanup

**⚠️ IMPORTANT:** To avoid unexpected charges, clean up all resources after completing the lab.

### Step 1: Delete Knowledge Base

1. Go to Bedrock Console:
   - https://console.aws.amazon.com/bedrock
   - Click **"Knowledge bases"**
   
2. Delete Knowledge Base:
   - Select your knowledge base (`my-rag-knowledge-base`)
   - Click **"Delete"**
   - Type the knowledge base name to confirm
   - Click **"Delete"**

---

### Step 2: Delete OpenSearch Serverless Collection

1. Go to OpenSearch Console:
   - https://console.aws.amazon.com/aos
   - Click **"Serverless"** → **"Collections"**

2. Delete Collection:
   - Select `bedrock-kb-collection`
   - Click **"Delete"**
   - Type `delete` to confirm
   - Click **"Delete"**

---

### Step 3: Delete OpenSearch Policies

1. **Delete Data Access Policy:**
   - Go to: Serverless → Security → Data access policies
   - Select `bedrock-kb-data-access-policy`
   - Click **"Delete"**

2. **Delete Network Policy:**
   - Go to: Serverless → Security → Network policies
   - Select `bedrock-kb-network-policy`
   - Click **"Delete"**

3. **Delete Encryption Policy:**
   - Go to: Serverless → Security → Encryption policies
   - Select `bedrock-kb-encryption-policy`
   - Click **"Delete"**

---

### Step 4: Empty and Delete S3 Bucket

1. Go to S3 Console:
   - https://console.aws.amazon.com/s3

2. Empty Bucket:
   - Click on your bucket name
   - Click **"Empty"**
   - Type `permanently delete` to confirm
   - Click **"Empty"**

3. Delete Bucket:
   - Go back to S3 bucket list
   - Select your bucket
   - Click **"Delete"**
   - Type the bucket name to confirm
   - Click **"Delete bucket"**

---

### Step 5: Delete IAM Role

1. Go to IAM Console:
   - https://console.aws.amazon.com/iam
   - Click **"Roles"**

2. Delete Role:
   - Search for `BedrockKnowledgeBaseRole`
   - Select the role
   - Click **"Delete"**
   - Type the role name to confirm
   - Click **"Delete"**

---

### Step 6: Verify Cleanup

**Checklist:**
- ✅ Knowledge Base deleted
- ✅ OpenSearch collection deleted
- ✅ OpenSearch policies deleted (all 3)
- ✅ S3 bucket emptied and deleted
- ✅ IAM role deleted

**Verification:**
- Go to each console and verify resources are gone
- No resources should appear in searches

---

### Optional: Clean Up Local Files

```bash
# Remove project directory
cd ..
rm -rf bedrock-rag-project

# Remove sample documents
rm -rf sample-docs

# Remove test results
rm -f chunk_test_results_*.json
```

---

## Additional Resources

### AWS Documentation
- [Amazon Bedrock User Guide](https://docs.aws.amazon.com/bedrock/)
- [Bedrock Knowledge Bases](https://docs.aws.amazon.com/bedrock/latest/userguide/knowledge-base.html)
- [Bedrock Agents](https://docs.aws.amazon.com/bedrock/latest/userguide/agents.html)
- [OpenSearch Serverless](https://docs.aws.amazon.com/opensearch-service/latest/developerguide/serverless.html)

### LangChain Resources
- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [LangChain AWS Integration](https://python.langchain.com/docs/integrations/platforms/aws)
- [Retrieval QA Chains](https://python.langchain.com/docs/use_cases/question_answering/)

### RAG Concepts
- [RAG Paper (Lewis et al.)](https://arxiv.org/abs/2005.11401)
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://ai.meta.com/research/publications/retrieval-augmented-generation-for-knowledge-intensive-nlp-tasks/)

### Best Practices
- [AWS Well-Architected Framework - ML Lens](https://docs.aws.amazon.com/wellarchitected/latest/machine-learning-lens/welcome.html)
- [Bedrock Best Practices](https://docs.aws.amazon.com/bedrock/latest/userguide/best-practices.html)

### Community
- [AWS re:Post - Bedrock](https://repost.aws/tags/TA4IHvpzj4TgK_1P_chZdJKA/amazon-bedrock)
- [LangChain Discord](https://discord.gg/langchain)

---

## What You've Learned

Congratulations! You've completed the AWS Bedrock Knowledge Base with RAG implementation lab. You now know how to:

✅ **Setup & Configuration**
- Enable Bedrock model access
- Create and configure Knowledge Bases
- Setup OpenSearch Serverless for vector storage
- Configure IAM roles and permissions

✅ **Document Processing**
- Upload documents to S3
- Trigger document ingestion and embedding
- Verify successful processing

✅ **RAG Implementation**
- Implement RAG pipeline with LangChain
- Generate embeddings using Titan
- Retrieve relevant documents
- Generate contextual responses with Claude

✅ **Performance Testing**
- Test retrieval with different queries
- Measure latency and relevance
- Analyze chunk size impact
- Optimize for your use case

✅ **Agent Integration**
- Create agents with RAG capabilities
- Implement multi-step reasoning
- Build complex workflows
- Track conversation history

---

## Next Steps

### Intermediate Projects
1. **Multi-modal RAG:** Add image understanding to your knowledge base
2. **Custom Chunking:** Implement document-specific chunking strategies
3. **Hybrid Search:** Combine keyword and semantic search
4. **Caching Layer:** Add Redis for query caching

### Advanced Projects
1. **Fine-tuning:** Customize foundation models with your data
2. **Guardrails:** Implement content filtering and safety checks
3. **Production Deployment:** Deploy with API Gateway and Lambda
4. **Monitoring:** Add CloudWatch metrics and alerting

### Real-world Applications
- Customer support chatbots
- Internal documentation search
- Research assistant tools
- Code documentation Q&A

---

## Feedback

Found an issue or have suggestions? Please provide feedback through:
- GitHub Issues on this repository
- AWS Support (for service-specific issues)
- Your training coordinator

---

**Version:** 2.0  
**Last Updated:** February 2026  
**Maintained by:** AWS Training Team
