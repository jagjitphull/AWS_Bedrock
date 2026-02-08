# AWS Bedrock Knowledge Base with RAG Implementation

## Hands-On Lab: Building a Production-Ready RAG Pipeline

**Estimated Duration:** 90-120 minutes  
**Difficulty Level:** Intermediate  
**Last Updated:** February 2026  
**Version:** 3.0 (Updated for AWS Bedrock 2026 features)

[![AWS](https://img.shields.io/badge/AWS-Bedrock-orange)](https://aws.amazon.com/bedrock/)
[![Python](https://img.shields.io/badge/Python-3.9+-blue)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3+-green)](https://python.langchain.com/)

---

## 📋 Table of Contents

- [Overview](#overview)
- [What's New in 2026](#whats-new-in-2026)
- [Prerequisites](#prerequisites)
- [Architecture Overview](#architecture-overview)
- [Lab 1: Setup AWS Bedrock Knowledge Base](#lab-1-setup-aws-bedrock-knowledge-base)
- [Lab 2: Create and Upload Documents](#lab-2-create-and-upload-documents)
- [Lab 3: Configure Embeddings](#lab-3-configure-embeddings)
- [Lab 4: Implement RAG with LangChain](#lab-4-implement-rag-with-langchain)
- [Lab 5: Test and Optimize Performance](#lab-5-test-and-optimize-performance)
- [Lab 6: Advanced Agent Integration](#lab-6-advanced-agent-integration)
- [Troubleshooting](#troubleshooting)
- [Cleanup](#cleanup)
- [Additional Resources](#additional-resources)

---

## 🎯 Overview

This hands-on lab guides you through building a complete **Retrieval-Augmented Generation (RAG)** system using AWS Bedrock Knowledge Base, embeddings, and LangChain. You'll learn how to:

- ✅ Create and configure an AWS Bedrock Knowledge Base
- ✅ Generate embeddings from your documents
- ✅ Implement RAG patterns using LangChain
- ✅ Optimize retrieval performance
- ✅ Integrate advanced agent capabilities

### What is RAG?

**Retrieval-Augmented Generation** combines information retrieval with language generation. Instead of relying solely on the model's training data, RAG retrieves relevant information from your knowledge base and uses it to generate more accurate, contextual responses.

### What You'll Build

A production-ready RAG pipeline that can answer questions based on your custom documents, with optimized retrieval and agent integration for complex workflows.

---

## ✨ What's New in 2026

AWS Bedrock has received major updates since 2025. This tutorial reflects all current best practices:

### 🔄 Simplified Model Access (October 2025)

- ✅ **Automatic Model Enablement** - No manual activation required for most models
- ✅ **Retired "Model Access" Page** - Streamlined authentication process
- ✅ **Faster Setup** - Models available immediately on first invocation
- ⚠️ **Exception:** Anthropic models require one-time use case form submission

### 🆕 Enhanced Features

**1. Multimodal Support**
- Process images (JPEG/PNG up to 3.75MB) alongside text
- Extract insights from charts, diagrams, and figures
- Supported models: Titan Multimodal Embeddings G1, Cohere Embed v3

**2. Additional Vector Store Options**
- 🆕 Amazon S3 Vectors (preview) - Fully managed vector storage
- Expanded support: MongoDB, Pinecone, Redis Enterprise Cloud, Neptune Analytics

**3. Enhanced Parsing**
- 🆕 Bedrock Data Automation (BDA) - Advanced document parsing
- Foundation Model Parsers - Use Claude or other FMs for parsing
- Improved table and figure extraction

**4. Reranking Models**
- Apply reranking after retrieval for improved relevance
- Better precision for top results across text and multimedia

**5. Structured Data Support**
- Connect to SQL databases directly
- Natural language to SQL query conversion

### 📚 This Tutorial Covers

We'll focus on core RAG implementation with:
- Text-based documents (TXT, PDF, MD)
- Amazon Titan Embeddings v2 (latest)
- OpenSearch Serverless vector store
- Claude 3.5 Sonnet for generation
- LangChain 0.3+ integration

> **Want multimodal or structured data?** See [Additional Resources](#additional-resources) for advanced guides.

---

## 📋 Prerequisites

### Required Tools

- **AWS Account** with appropriate permissions
- **AWS CLI** (v2.x or higher) installed and configured
- **Python 3.9+** installed
- **pip** (Python package manager)
- **Git** (for version control)
- **Text editor** or IDE (VS Code recommended)

### AWS Permissions Required

Your IAM user/role needs these permissions (least-privilege approach):

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "BedrockModelAccess",
      "Effect": "Allow",
      "Action": [
        "bedrock:InvokeModel",
        "bedrock:InvokeModelWithResponseStream",
        "bedrock:Retrieve",
        "bedrock:RetrieveAndGenerate"
      ],
      "Resource": [
        "arn:aws:bedrock:*::foundation-model/amazon.titan-embed-text-v*",
        "arn:aws:bedrock:*::foundation-model/anthropic.claude*"
      ]
    },
    {
      "Sid": "BedrockKnowledgeBaseAccess",
      "Effect": "Allow",
      "Action": [
        "bedrock:CreateKnowledgeBase",
        "bedrock:GetKnowledgeBase",
        "bedrock:ListKnowledgeBases",
        "bedrock:UpdateKnowledgeBase",
        "bedrock:DeleteKnowledgeBase",
        "bedrock:CreateDataSource",
        "bedrock:GetDataSource",
        "bedrock:ListDataSources",
        "bedrock:StartIngestionJob",
        "bedrock:GetIngestionJob"
      ],
      "Resource": "*"
    },
    {
      "Sid": "S3Access",
      "Effect": "Allow",
      "Action": [
        "s3:CreateBucket",
        "s3:ListBucket",
        "s3:GetObject",
        "s3:PutObject",
        "s3:DeleteObject",
        "s3:PutBucketVersioning"
      ],
      "Resource": [
        "arn:aws:s3:::bedrock-kb-docs-*",
        "arn:aws:s3:::bedrock-kb-docs-*/*"
      ]
    },
    {
      "Sid": "IAMRoleAccess",
      "Effect": "Allow",
      "Action": [
        "iam:CreateRole",
        "iam:AttachRolePolicy",
        "iam:PassRole",
        "iam:GetRole",
        "iam:CreatePolicy"
      ],
      "Resource": "*"
    },
    {
      "Sid": "OpenSearchServerlessAccess",
      "Effect": "Allow",
      "Action": [
        "aoss:*",
        "opensearchserverless:*"
      ],
      "Resource": "*"
    }
  ]
}
```

### 📦 Install Required Python Packages

**Create requirements.txt:**

```text
# AWS Bedrock RAG Requirements
# Updated February 2026

boto3>=1.34.0
langchain>=0.3.0
langchain-aws>=1.2.2
langchain-community>=0.3.0
opensearch-py>=2.4.0
requests>=2.31.0
requests-aws4auth>=1.2.0
```

**For Unix/Linux/Mac (Bash):**

```bash
# Create a virtual environment
python3 -m venv bedrock-rag-env
source bedrock-rag-env/bin/activate

# Install required packages
pip install --upgrade pip
pip install -r requirements.txt

# Verify installation
pip show langchain-aws  # Should be 1.2.2 or higher
```

**For Windows (PowerShell):**

```powershell
# Create a virtual environment
python -m venv bedrock-rag-env
.\bedrock-rag-env\Scripts\Activate.ps1

# Install required packages
pip install --upgrade pip
pip install -r requirements.txt

# Verify installation
pip show langchain-aws  # Should be 1.2.2 or higher
```

### 🔧 Verify AWS CLI Configuration

```bash
# Check AWS CLI installation
aws --version

# Verify credentials
aws sts get-caller-identity

# Check Bedrock availability in your region
aws bedrock list-foundation-models --region us-east-1 | head -20
```

**Expected Output:**

```json
{
    "UserId": "AIDAXXXXXXXXXXXXXXXXX",
    "Account": "123456789012",
    "Arn": "arn:aws:iam::123456789012:user/your-username"
}
```

### 📊 Model Versions and Lifecycle

AWS Bedrock models follow a lifecycle:
- **Active:** Current production models (12+ months support guaranteed)
- **Legacy:** Older versions (6+ months before EOL)
- **EOL:** End-of-life (no longer available)

**Models used in this tutorial (February 2026):**
- `amazon.titan-embed-text-v2:0` - Active ✅
- `anthropic.claude-3-5-sonnet-20240620-v1:0` - Active ✅

**To check current model status:**

```bash
aws bedrock list-foundation-models --region us-east-1 | \
  jq '.modelSummaries[] | select(.modelId | contains("titan-embed") or contains("claude")) | {modelId, status: .modelLifecycle.status}'
```

> **For production:** Always verify model lifecycle status and plan migrations before EOL dates.  
> See: https://docs.aws.amazon.com/bedrock/latest/userguide/model-lifecycle.html

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Your Documents                           │
│                  (PDF, TXT, MD, DOCX)                       │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
          ┌──────────────────────┐
          │     S3 Bucket        │
          │  (Document Storage)  │
          └──────────┬───────────┘
                     │
                     ▼
          ┌──────────────────────────────┐
          │  Bedrock Knowledge Base      │
          │  - Document Chunking         │
          │  - Titan Embeddings v2       │
          │  - Metadata Extraction       │
          └──────────┬───────────────────┘
                     │
                     ▼
          ┌──────────────────────────────┐
          │ OpenSearch Serverless        │
          │  (Vector Database)           │
          │  - 1536-dim vectors          │
          │  - HNSW indexing             │
          └──────────┬───────────────────┘
                     │
                     ▼
          ┌──────────────────────────────┐
          │     LangChain RAG Pipeline   │
          │  - Query Processing          │
          │  - Context Retrieval         │
          │  - Claude 3.5 Generation     │
          └──────────┬───────────────────┘
                     │
                     ▼
          ┌──────────────────────────────┐
          │    Agent Framework           │
          │  - Multi-step Reasoning      │
          │  - Tool Integration          │
          └──────────────────────────────┘
```

**Data Flow:**
1. **Ingestion:** Documents uploaded to S3
2. **Processing:** Bedrock chunks and extracts text
3. **Embedding:** Titan creates 1536-dim vectors
4. **Storage:** Vectors stored in OpenSearch
5. **Retrieval:** Semantic search finds relevant chunks
6. **Generation:** Claude generates contextual answers

---

## 🚀 Lab 1: Setup AWS Bedrock Knowledge Base

### Step 1.1: Verify Bedrock Model Access

> **🎉 MAJOR UPDATE (October 2025):** AWS Bedrock now provides automatic access to models by default. The "Model Access" page has been retired. Most models are ready to use immediately!

#### What Changed

- ❌ No more "Model Access" page
- ❌ No more manual model enablement for most models  
- ❌ No more waiting 2-5 minutes for activation
- ✅ Models available immediately on first invocation
- ✅ Only Anthropic requires one-time use case form
- ✅ AWS Marketplace models auto-enabled on first use

#### Steps

**1. Navigate to AWS Bedrock Console:**

- Go to https://console.aws.amazon.com/bedrock
- Select your region (recommend: **us-east-1** or **us-west-2**)

**2. For Anthropic Models (Claude) - One-Time Requirement:**

Anthropic requires first-time customers to submit a use case form before using their models.

**Option A - Via Playground (Easiest):**

1. Click **"Playgrounds"** → **"Text"** in the left navigation
2. Click **"Select model"**
3. Choose any Anthropic Claude model (e.g., "Claude 3.5 Sonnet")
4. A use case form will appear automatically
5. Fill in:
   - **Use case description:** Enter your intended use (e.g., "Document Q&A for internal knowledge base")
   - Accept terms and conditions
6. Click **"Submit"**
7. ✅ Access is granted **immediately** after submission

**Option B - Via AWS CLI (For Automation):**

```bash
aws bedrock put-use-case-for-model-access \
  --model-id anthropic.claude-3-5-sonnet-20240620-v1:0 \
  --use-case "Document Q&A system for training purposes" \
  --region us-east-1
```

**3. Verify Model Availability:**

All other models (Amazon Titan, Cohere, AI21, Meta Llama, etc.) are automatically available.

**Test with AWS CLI:**

```bash
# List all available models
aws bedrock list-foundation-models --region us-east-1

# Filter for models we'll use
aws bedrock list-foundation-models --region us-east-1 | \
  jq '.modelSummaries[] | select(.modelId | contains("titan-embed") or contains("claude-3")) | {modelId, status: .modelLifecycle.status}'
```

**Expected Output:**

```json
{
  "modelId": "amazon.titan-embed-text-v2:0",
  "status": "ACTIVE"
}
{
  "modelId": "anthropic.claude-3-5-sonnet-20240620-v1:0",
  "status": "ACTIVE"
}
```

**4. Test Model Access (Optional but Recommended):**

Create a quick test script to verify access:

```python
#!/usr/bin/env python3
"""Test Bedrock model access before proceeding."""

import boto3
import json

def test_model_access():
    """Verify models are accessible."""
    bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-east-1')
    
    print("\n" + "=" * 60)
    print("Testing Bedrock Model Access")
    print("=" * 60)
    
    # Test Titan Embeddings
    print("\n🧪 Testing Titan Embeddings v2...")
    try:
        response = bedrock_runtime.invoke_model(
            modelId='amazon.titan-embed-text-v2:0',
            body=json.dumps({"inputText": "test"}),
            contentType='application/json',
            accept='application/json'
        )
        print("✅ Titan Embeddings v2: Accessible")
    except Exception as e:
        print(f"❌ Titan Embeddings: {str(e)}")
        if "agreement" in str(e).lower():
            print("   → Try invoking once in the console first")
    
    # Test Claude
    print("\n🧪 Testing Claude 3.5 Sonnet...")
    try:
        response = bedrock_runtime.invoke_model(
            modelId='anthropic.claude-3-5-sonnet-20240620-v1:0',
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "Hi"}]
            }),
            contentType='application/json',
            accept='application/json'
        )
        print("✅ Claude 3.5 Sonnet: Accessible")
    except Exception as e:
        if "use case" in str(e).lower():
            print("❌ Claude: Use case form not submitted")
            print("   → Go to Bedrock Console → Playgrounds → Text")
            print("   → Select Claude model and submit the form")
        else:
            print(f"❌ Claude: {str(e)}")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    test_model_access()
```

Save as `test_model_access.py` and run:

```bash
python test_model_access.py
```

> **⚡ Quick Start:** Models are now ready! You can proceed directly to creating your S3 bucket and Knowledge Base.

---

### Step 1.2: Create S3 Bucket for Documents

**1. Navigate to S3 Console:**

- Go to https://console.aws.amazon.com/s3
- Click **"Create bucket"**

**2. Configure Bucket:**

- **Bucket name:** `bedrock-kb-docs-[your-unique-id]` 
  - Example: `bedrock-kb-docs-20260208`
  - Must be globally unique across all AWS accounts
- **AWS Region:** Select your region (e.g., `us-east-1`)
- **Block Public Access:** Keep all boxes checked ✅ (recommended)
- Leave other settings as default

**3. Create Bucket:**

- Scroll to bottom and click **"Create bucket"**

**4. Enable Versioning (Recommended):**

- Click on your newly created bucket name
- Go to **"Properties"** tab
- Scroll to **"Bucket Versioning"**
- Click **"Edit"** → Select **"Enable"** → Click **"Save changes"**

**5. Create Documents Folder:**

- Inside your bucket, click **"Create folder"**
- **Folder name:** `documents`
- Click **"Create folder"**

**📝 Save your bucket name - you'll need it throughout this lab!**

---

### Step 1.3: Create IAM Role for Knowledge Base

**1. Navigate to IAM Console:**

- Go to https://console.aws.amazon.com/iam
- Click **"Roles"** in left navigation
- Click **"Create role"**

**2. Select Trusted Entity:**

- **Trusted entity type:** Select **"AWS service"**
- **Use case:** Select **"Bedrock"**
- Click **"Next"**

**3. Create and Attach Policy:**

Click **"Create policy"** (opens in new tab):

- Click **"JSON"** tab
- Paste this policy (replace `YOUR-BUCKET-NAME` with your actual bucket name):

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "S3ReadAccess",
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
      "Sid": "BedrockEmbeddingAccess",
      "Effect": "Allow",
      "Action": [
        "bedrock:InvokeModel"
      ],
      "Resource": [
        "arn:aws:bedrock:*::foundation-model/amazon.titan-embed-text-v*",
        "arn:aws:bedrock:*::foundation-model/cohere.embed-*"
      ]
    },
    {
      "Sid": "OpenSearchServerlessAccess",
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
- **Description:** `Allows Bedrock Knowledge Base to access S3 and OpenSearch`
- Click **"Create policy"**
- Close the policy tab and return to role creation

**4. Attach the Policy:**

- Click the refresh button (🔄) next to "Create policy"
- Search for `BedrockKnowledgeBasePolicy`
- Check the box next to it
- Click **"Next"**

**5. Name and Create Role:**

- **Role name:** `BedrockKnowledgeBaseRole`
- **Description:** `Role for Bedrock Knowledge Base to access S3 and OpenSearch`
- Click **"Create role"**

**📝 Save the role name - you'll need it when creating the Knowledge Base!**

---

### Step 1.4: Create OpenSearch Serverless Collection

> **Important:** OpenSearch Serverless requires security policies to be created **before** the collection.

#### Step 1.4.1: Create Encryption Security Policy

**1. Navigate to OpenSearch Service Console:**

- Go to https://console.aws.amazon.com/aos
- In the left navigation, expand **"Serverless"**
- Click **"Security"** → **"Encryption policies"**

**2. Create Encryption Policy:**

- Click **"Create encryption policy"**
- **Policy name:** `bedrock-kb-encryption-policy`
- **Policy definition:**
  - Click **"Add rule"**
  - **Resource type:** Select `Collections`
  - **Collection name pattern:** `bedrock-kb-collection`
- **Encryption:**
  - Select **"Use AWS managed key"**
- Click **"Create"**

**✅ Expected Result:** Policy shows "Active" status

---

#### Step 1.4.2: Create Network Security Policy

**1. In OpenSearch Service Console:**

- Click **"Serverless"** → **"Security"** → **"Network policies"**

**2. Create Network Policy:**

- Click **"Create network policy"**
- **Policy name:** `bedrock-kb-network-policy`
- **Description:** `Network access policy for Bedrock Knowledge Base` (optional)
- **Access type:**
  - Select **"Public"** (for this lab)
  - ⚠️ *In production, select "VPC" and specify your VPCs*
- **Policy rules:**
  - Click **"Add rule"**
  - **Resource type:** Select `Collections`
  - **Collection name pattern:** `bedrock-kb-collection`
- Click **"Create"**

**✅ Expected Result:** Policy shows "Active" status

---

#### Step 1.4.3: Create the Collection

**1. In OpenSearch Service Console:**

- Click **"Serverless"** → **"Collections"**
- Click **"Create collection"**

**2. Configure Collection:**

- **Collection name:** `bedrock-kb-collection`
- **Description:** `Vector database for Bedrock Knowledge Base` (optional)
- **Collection type:** Select **"Vector search"**

**3. Configure Security:**

- **Encryption:**
  - **Select existing encryption policy:** Choose `bedrock-kb-encryption-policy`
- **Network:**
  - **Select existing network policy:** Choose `bedrock-kb-network-policy`

**4. Configure Capacity:**

- Leave at default settings (OpenSearch will auto-scale)

**5. Review and Create:**

- Review all settings
- Click **"Create"**

**⏳ Wait Time:** Collection takes **3-5 minutes** to provision

---

#### Step 1.4.4: Wait for Collection to Become ACTIVE

**1. Monitor Status:**

- Stay on the Collections page
- Click the refresh button (🔄) periodically
- Wait until **Status** changes from "Creating" to **"Active"** (green)

**2. Save Collection Endpoint:**

- Once active, click on your collection name `bedrock-kb-collection`
- Copy the **Collection endpoint** - it looks like:
  - Example: `https://abc123xyz.us-east-1.aoss.amazonaws.com`
  - **You'll need this for the vector index script!**

**Visual Indicator:**
```
Status: Creating ⏳  →  Status: Active ✅
```

---

#### Step 1.4.5: Create Data Access Policy

**1. In OpenSearch Service Console:**

- Click **"Serverless"** → **"Security"** → **"Data access policies"**

**2. Create Data Access Policy:**

- Click **"Create access policy"**
- **Policy name:** `bedrock-kb-data-access-policy`
- **Description:** `Access policy for Bedrock Knowledge Base` (optional)

**3. Add Policy Rules:**

**Rule 1 - Collection Permissions:**
- Click **"Add resource"**
- **Resource type:** Select `Collections`
- **Collections:** Select or enter `bedrock-kb-collection`
- **Grant permissions:** ✅ Select **ALL** collection permissions:
  - `aoss:CreateCollectionItems`
  - `aoss:DeleteCollectionItems`
  - `aoss:UpdateCollectionItems`
  - `aoss:DescribeCollectionItems`

**Rule 2 - Index Permissions:**
- Click **"Add another resource"**
- **Resource type:** Select `Indexes`
- **Index pattern:** Enter `bedrock-kb-collection/*`
- **Grant permissions:** ✅ Select **ALL** index permissions:
  - `aoss:CreateIndex`
  - `aoss:DeleteIndex`
  - `aoss:UpdateIndex`
  - `aoss:DescribeIndex`
  - `aoss:ReadDocument`
  - `aoss:WriteDocument`

**4. Add Principals (Who Can Access):**

- Scroll down to **"Policy principals"** section
- Click **"Add principals"**
- **IAM role:** Enter `BedrockKnowledgeBaseRole`
- The full ARN should auto-complete like:
  - `arn:aws:iam::123456789012:role/BedrockKnowledgeBaseRole`
- **Also add your own user/role** if you want to manage the collection

> **💡 How to find your Account ID:**
> - Click your username in top-right corner
> - Your 12-digit Account ID is displayed
> - Or go to: https://console.aws.amazon.com/billing/

**5. Review and Create:**

- Review all settings
- Click **"Create"**

**✅ Expected Result:** Policy shows "Active" status

---

#### Step 1.4.6: Create Vector Index

The vector index stores document embeddings and enables similarity search. This requires running a Python script.

**Prerequisites:**

```bash
# Install required Python packages (one-time setup)
pip install opensearch-py boto3 requests requests-aws4auth
```

**Create the Script:**

Save this as `create_vector_index.py`:

```python
#!/usr/bin/env python3
"""
Create vector index in OpenSearch Serverless for Bedrock Knowledge Base
Updated for 2026 - Supports Titan Embeddings v2
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
    """Create vector index with proper configuration."""
    
    # Configuration - UPDATE THESE VALUES
    region = 'us-east-1'  # Change to your region
    collection_name = 'bedrock-kb-collection'
    index_name = 'bedrock-kb-index'
    vector_dimensions = 1536  # For Amazon Titan Embeddings G1 and v2
    
    print("\n" + "=" * 70)
    print("Creating Vector Index for Bedrock Knowledge Base")
    print("=" * 70)
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
        
        print("\n" + "=" * 70)
        print("✅ SUCCESS! Vector Index Created")
        print("=" * 70)
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
        print("=" * 70)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error creating index: {e}")
        
        if "AccessDeniedException" in str(e):
            print("\n💡 PERMISSION ERROR!")
            print("   Make sure you created the data access policy (Step 1.4.5)")
            print("   The policy must include your IAM user or role")
        
        return False

def main():
    """Main execution."""
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

**Run the Script:**

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

1. Make sure AWS CLI is configured: `aws configure`
2. Save the Python script as `create_vector_index.py`
3. Run:
   ```bash
   python3 create_vector_index.py
   ```

**Expected Output:**

```
======================================================================
Creating Vector Index for Bedrock Knowledge Base
======================================================================

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

======================================================================
✅ SUCCESS! Vector Index Created
======================================================================

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

======================================================================
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

**1. Navigate to Bedrock Console:**

- Go to https://console.aws.amazon.com/bedrock
- Click **"Knowledge bases"** in the left menu
- Click **"Create knowledge base"**

**2. Provide Knowledge Base Details (Step 1):**

- **Name:** `my-rag-knowledge-base`
- **Description:** `RAG knowledge base for document Q&A` (optional)
- **IAM Role:**
  - Select **"Use an existing service role"**
  - Choose: `BedrockKnowledgeBaseRole`
- **Tags:** (Optional) Add tags for organization
- Click **"Next"**

**3. Configure Data Source (Step 2):**

- **Data source name:** `s3-documents`
- **S3 URI:**
  - Click **"Browse S3"**
  - Select your bucket (e.g., `bedrock-kb-docs-20260208`)
  - Select the `documents` folder
  - Or manually enter: `s3://your-bucket-name/documents/`

- **Chunking and parsing settings:**
  - Leave as **"Default chunking"** (recommended for beginners)
  - **Advanced Options** (Optional, for experimentation):
    - Fixed size chunking: ~300 tokens with 20% overlap
    - Hierarchical chunking: For documents with clear structure
    - Semantic chunking: For varied content types
    - Custom chunking: Via Lambda function

- **🆕 NEW: Advanced parsing options** (Optional):
  - Default parser (recommended)
  - Bedrock Data Automation (for complex documents with tables/images)
  - Foundation model parser (Claude or other FMs)

- Click **"Next"**

**4. Select Embeddings Model and Configure Vector Store (Step 3):**

**Embeddings Model:**
- **Select embeddings model:** `Titan Embeddings G1 - Text v2` ✅ **UPDATED**
- **Alternative:** `Cohere Embed English v3` (better multilingual support)
- **Dimensions:** 1536 (Titan v2) or configurable (Cohere)
- **🆕 Embeddings type:** 
  - Float32 (default, more precise) ← Recommended
  - Binary (less precise, more cost-effective)

**Vector Database:**
- **Vector database:** Select **"Amazon OpenSearch Serverless"**
- **Alternative options:**
  - 🆕 Amazon S3 Vectors (preview, fully managed)
  - Amazon Aurora (PostgreSQL)
  - MongoDB Atlas
  - Pinecone
  - Redis Enterprise Cloud

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

> **💡 Tip:** Copy-paste these field names from the script output to avoid typos!

- Click **"Next"**

**5. Review and Create (Step 4):**

- Review all your settings
- Verify the field names are correct
- Verify embedding model is Titan v2
- Click **"Create knowledge base"**
- Wait 2-3 minutes for creation to complete

**6. Save Your Knowledge Base ID:**

- Once created, you'll see your Knowledge Base details page
- Note the **Knowledge base ID** (looks like: `ABC123XYZ` or `kb-XXXXXXXXXXXXX`)
- **You'll need this ID for testing and integration!**

**Expected Result:** Knowledge Base status shows "Active" (green)

---

**✅ Lab 1 Checkpoint - You should now have:**

- ✅ Bedrock models accessible (Titan Embeddings v2, Claude 3.5 Sonnet)
- ✅ S3 bucket created with `documents` folder
- ✅ IAM role configured (BedrockKnowledgeBaseRole)
- ✅ OpenSearch security policies created (encryption, network, data access)
- ✅ OpenSearch collection active (bedrock-kb-collection)
- ✅ Vector index created (bedrock-kb-index)
- ✅ Knowledge base created with correct field mappings
- ✅ Knowledge Base ID saved

---

## 📄 Lab 2: Create and Upload Documents

### Step 2.1: Prepare Sample Documents

Create a directory for your documents and add sample content.

**Create project structure:**

```bash
mkdir -p sample-docs
cd sample-docs
```

**Create Sample Document 1: AWS Overview**

Save as `aws-overview.txt`:

```text
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
```

**Create Sample Document 2: Bedrock Information**

Save as `bedrock-intro.txt`:

```text
Amazon Bedrock Overview

Amazon Bedrock is a fully managed service that offers a choice of high-performing foundation models (FMs) from leading AI companies like AI21 Labs, Anthropic, Cohere, Meta, Stability AI, and Amazon via a single API. With Bedrock, you can easily experiment with and evaluate top FMs for your use case, privately customize them with your data, and build agents that execute tasks using your enterprise systems and data sources.

Key Features:
1. Choice of Leading Foundation Models
   - Anthropic Claude (multiple versions including Claude 3.5 and Claude 4)
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
   - Support for various document formats including multimodal content

5. Guardrails
   - Define content filters
   - Control model responses
   - Ensure responsible AI usage
   - Block harmful content

Bedrock Knowledge Bases allow you to build RAG applications without managing vector databases. The service handles:
- Document ingestion from S3
- Chunking and text extraction
- Embedding generation with Titan v2 or Cohere
- Vector storage in OpenSearch Serverless, S3 Vectors, or other supported databases
- Retrieval and ranking with optional reranking models

Use Cases:
- Conversational AI and chatbots
- Text summarization and generation
- Image generation and editing
- Search and question-answering
- Code generation and explanation
- Content personalization
```

**Create Sample Document 3: RAG Concepts**

Save as `rag-concepts.txt`:

```text
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
   - Examples: OpenSearch, Pinecone, Chroma, FAISS, S3 Vectors

3. Retrieval Mechanism
   - Query embedding: Converting user questions to vectors
   - Similarity search: Finding nearest neighbors in vector space
   - Ranking: Ordering results by relevance
   - Reranking: Optional second-pass ranking for improved precision
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
- Typical sizes: 256-1024 tokens

Semantic Chunking:
- Split based on meaning and context
- Preserves logical units (paragraphs, sections)
- More complex but better quality
- Ideal for varied content

Hierarchical Chunking:
- Creates parent-child relationships
- Enables multi-level retrieval
- Better context preservation

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
- Context Recall: Are all relevant chunks retrieved?
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
9. Use reranking models for improved precision
10. Consider multimodal content when applicable
```

**Quick create script (Bash):**

```bash
cd sample-docs

# Create all three files at once
cat > aws-overview.txt << 'EOF'
[Paste content from above]
EOF

cat > bedrock-intro.txt << 'EOF'
[Paste content from above]
EOF

cat > rag-concepts.txt << 'EOF'
[Paste content from above]
EOF

echo "✅ Sample documents created successfully!"
ls -lh
```

---

### Step 2.2: Upload Documents to S3

**Using AWS Console:**

1. Navigate to S3 Console: https://console.aws.amazon.com/s3
2. Click on your bucket (e.g., `bedrock-kb-docs-20260208`)
3. Navigate to the `documents` folder
4. Click **"Upload"**
5. Click **"Add files"**
6. Select all three `.txt` files:
   - `aws-overview.txt`
   - `bedrock-intro.txt`
   - `rag-concepts.txt`
7. Click **"Upload"**
8. Wait for upload to complete
9. Click **"Close"**

**Using AWS CLI:**

```bash
# Navigate to your documents directory
cd sample-docs

# Set your bucket name
BUCKET_NAME="bedrock-kb-docs-20260208"  # Replace with your bucket name

# Upload all documents
aws s3 cp aws-overview.txt s3://${BUCKET_NAME}/documents/
aws s3 cp bedrock-intro.txt s3://${BUCKET_NAME}/documents/
aws s3 cp rag-concepts.txt s3://${BUCKET_NAME}/documents/

# Verify upload
aws s3 ls s3://${BUCKET_NAME}/documents/
```

**Expected Output:**

```
upload: ./aws-overview.txt to s3://bedrock-kb-docs-20260208/documents/aws-overview.txt
upload: ./bedrock-intro.txt to s3://bedrock-kb-docs-20260208/documents/bedrock-intro.txt
upload: ./rag-concepts.txt to s3://bedrock-kb-docs-20260208/documents/rag-concepts.txt

2026-02-08 10:30:45       2156 aws-overview.txt
2026-02-08 10:30:46       1843 bedrock-intro.txt
2026-02-08 10:30:47       3245 rag-concepts.txt
```

---

### Step 2.3: Sync Knowledge Base with S3

After uploading documents, trigger the knowledge base to ingest and process them.

**Using AWS Console:**

1. Go to Bedrock Console: https://console.aws.amazon.com/bedrock
2. Click **"Knowledge bases"** in left menu
3. Click on your knowledge base (`my-rag-knowledge-base`)
4. Click on the **"Data source"** tab
5. You should see your S3 data source (`s3-documents`)
6. Click the **checkbox** next to the data source name
7. Click **"Sync"** button at the top
8. Confirm by clicking **"Sync"** in the dialog

**Monitor Sync Progress:**

The "Sync status" column will show:
- `Starting` → `In progress` → `Completed`

**Timeline:**
- Starting: ~30 seconds
- Processing documents: ~3-5 minutes
- Creating embeddings: ~2-4 minutes
- **Total: ~5-10 minutes**

Click the **refresh button** (🔄) to update status.

**Verify Completion:**

Wait until "Sync status" shows **"Completed"** (green):
- Check "Documents synced" count - should show `3`
- Check "Last sync" timestamp - should be recent

**Visual Indicator:**
```
Sync Status: Starting ⏳ → In progress ⏳ → Completed ✅
Documents synced: 0 → 3
```

**Using AWS CLI (Alternative):**

```bash
# Set variables
KB_ID="your-knowledge-base-id"  # Replace with your KB ID
DATA_SOURCE_ID="your-data-source-id"  # Get from console

# Start ingestion job
aws bedrock-agent start-ingestion-job \
  --knowledge-base-id $KB_ID \
  --data-source-id $DATA_SOURCE_ID \
  --region us-east-1

# Monitor job status
aws bedrock-agent get-ingestion-job \
  --knowledge-base-id $KB_ID \
  --data-source-id $DATA_SOURCE_ID \
  --ingestion-job-id <JOB_ID> \
  --region us-east-1
```

---

**✅ Lab 2 Checkpoint - You should now have:**

- ✅ Three sample documents created
- ✅ Documents uploaded to S3 `documents` folder
- ✅ Data source sync completed successfully
- ✅ All 3 documents processed and embedded
- ✅ Embeddings stored in vector index
- ✅ Knowledge Base ready for querying

---

## 🔢 Lab 3: Configure Embeddings

### Understanding Embeddings

Embeddings are vector representations of text that capture semantic meaning. Similar texts have similar vectors, enabling semantic search.

**Titan Embeddings v2 Specifications:**
- **Model ID:** `amazon.titan-embed-text-v2:0`
- **Dimensions:** 1536 (same as v1 for compatibility)
- **Max Input Tokens:** 8,192
- **Supported Languages:** 100+ languages
- **Improvements over v1:**
  - Better multilingual support
  - Improved accuracy for technical content
  - Enhanced domain adaptation

### Step 3.1: Test Embedding Generation

Create a Python script to test embedding generation:

**File:** `test_embeddings.py`

```python
#!/usr/bin/env python3
"""Test Titan Embeddings v2 generation."""

import boto3
import json
import numpy as np

# Initialize Bedrock client
bedrock_runtime = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-1'
)

def generate_embedding(text, model_id='amazon.titan-embed-text-v2:0'):
    """Generate embedding for input text using Titan v2."""
    body = json.dumps({
        "inputText": text
    })
    
    response = bedrock_runtime.invoke_model(
        modelId=model_id,
        body=body,
        contentType='application/json',
        accept='application/json'
    )
    
    response_body = json.loads(response['body'].read())
    embedding = response_body.get('embedding')
    
    return embedding

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm_a = np.linalg.norm(vec1)
    norm_b = np.linalg.norm(vec2)
    return dot_product / (norm_a * norm_b)

def main():
    """Test embedding generation."""
    # Test with sample texts
    texts = [
        "What is AWS Lambda?",
        "How does serverless computing work?",
        "What is Amazon S3 used for?",
        "Tell me about object storage."
    ]
    
    print("\n" + "=" * 70)
    print("Testing Titan Embeddings v2")
    print("=" * 70)
    print("\nGenerating embeddings for sample texts...\n")
    
    embeddings = []
    for i, text in enumerate(texts, 1):
        print(f"{i}. Text: {text}")
        embedding = generate_embedding(text)
        embeddings.append(embedding)
        print(f"   Embedding dimensions: {len(embedding)}")
        print(f"   First 5 values: {[round(v, 4) for v in embedding[:5]]}")
        print()
    
    # Calculate similarity
    print("\n" + "=" * 70)
    print("Similarity Analysis")
    print("=" * 70)
    print(f"\nSimilarity between text 1 and 2 (related):")
    print(f"  {cosine_similarity(embeddings[0], embeddings[1]):.4f}")
    print(f"\nSimilarity between text 1 and 3 (unrelated):")
    print(f"  {cosine_similarity(embeddings[0], embeddings[2]):.4f}")
    print(f"\nSimilarity between text 3 and 4 (related):")
    print(f"  {cosine_similarity(embeddings[2], embeddings[3]):.4f}")
    
    print("\n" + "=" * 70)
    print("✅ Embedding generation test completed!")
    print("=" * 70)

if __name__ == "__main__":
    main()
```

**Run the script:**

```bash
python test_embeddings.py
```

**Expected Output:**

```
======================================================================
Testing Titan Embeddings v2
======================================================================

Generating embeddings for sample texts...

1. Text: What is AWS Lambda?
   Embedding dimensions: 1536
   First 5 values: [0.0234, -0.0156, 0.0891, -0.0423, 0.0567]

2. Text: How does serverless computing work?
   Embedding dimensions: 1536
   First 5 values: [0.0198, -0.0143, 0.0856, -0.0401, 0.0534]

3. Text: What is Amazon S3 used for?
   Embedding dimensions: 1536
   First 5 values: [-0.0123, 0.0234, -0.0456, 0.0678, -0.0234]

4. Text: Tell me about object storage.
   Embedding dimensions: 1536
   First 5 values: [-0.0145, 0.0267, -0.0489, 0.0701, -0.0256]

======================================================================
Similarity Analysis
======================================================================

Similarity between text 1 and 2 (related):
  0.8523

Similarity between text 1 and 3 (unrelated):
  0.6234

Similarity between text 3 and 4 (related):
  0.8912

======================================================================
✅ Embedding generation test completed!
======================================================================
```

**Interpretation:**
- Related texts (Lambda/serverless, S3/object storage) have high similarity (>0.85)
- Unrelated texts have lower similarity (~0.62)
- This demonstrates semantic understanding

---

### Step 3.2: Verify Knowledge Base Embeddings

Check that embeddings were created for your documents:

**File:** `verify_kb_embeddings.py`

```python
#!/usr/bin/env python3
"""Verify Knowledge Base embeddings and retrieval."""

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

def main():
    """Test Knowledge Base retrieval."""
    # Test queries
    test_queries = [
        "What is AWS Lambda?",
        "Explain RAG architecture components",
        "What are the key features of Amazon Bedrock?"
    ]
    
    print("\n" + "=" * 70)
    print(f"Testing Knowledge Base: {KB_ID}")
    print("=" * 70)
    
    for query in test_queries:
        print(f"\n📝 Query: {query}")
        print("-" * 70)
        
        try:
            results = query_knowledge_base(query, max_results=2)
            
            if not results:
                print("⚠️  No results found")
                continue
            
            for i, result in enumerate(results, 1):
                print(f"\nResult {i}:")
                print(f"  Score: {result['score']:.4f}")
                print(f"  Content: {result['content']['text'][:200]}...")
                if 'location' in result:
                    location = result['location'].get('s3Location', {})
                    print(f"  Source: {location.get('uri', 'N/A')}")
        
        except Exception as e:
            print(f"❌ Error: {str(e)}")
        
        print("-" * 70)
    
    print("\n" + "=" * 70)
    print("✅ Verification completed!")
    print("=" * 70)

if __name__ == "__main__":
    main()
```

**Run the script:**

```bash
# Set environment variable first (replace with your actual KB ID)
export KB_ID="your-actual-kb-id"
python verify_kb_embeddings.py
```

**Expected Output:**

```
======================================================================
Testing Knowledge Base: ABC123XYZ
======================================================================

📝 Query: What is AWS Lambda?
----------------------------------------------------------------------

Result 1:
  Score: 0.8234
  Content: Lambda: Serverless computing platform
DynamoDB: NoSQL database service

AWS operates in 32 geographic regions around the world with 102 Availability Zones...
  Source: s3://bedrock-kb-docs-20260208/documents/aws-overview.txt

Result 2:
  Score: 0.7456
  Content: Amazon Bedrock is a fully managed service that offers a choice of high-performing foundation models (FMs) from leading AI companies...
  Source: s3://bedrock-kb-docs-20260208/documents/bedrock-intro.txt
----------------------------------------------------------------------

📝 Query: Explain RAG architecture components
----------------------------------------------------------------------

Result 1:
  Score: 0.8912
  Content: RAG Architecture Components:

1. Document Processing Pipeline
   - Document ingestion: Loading documents from various sources
   - Text extraction: Converting documents to plain text...
  Source: s3://bedrock-kb-docs-20260208/documents/rag-concepts.txt

Result 2:
  Score: 0.7823
  Content: Bedrock Knowledge Bases allow you to build RAG applications without managing vector databases. The service handles:
- Document ingestion from S3...
  Source: s3://bedrock-kb-docs-20260208/documents/bedrock-intro.txt
----------------------------------------------------------------------

======================================================================
✅ Verification completed!
======================================================================
```

**What to Look For:**
- ✅ Relevance scores above 0.7 for related queries
- ✅ Correct source attribution (S3 URIs)
- ✅ Relevant content chunks returned
- ✅ Top result matches query intent

---

**✅ Lab 3 Checkpoint - You should see:**

- ✅ Documents uploaded to S3
- ✅ Ingestion job completed successfully
- ✅ Embeddings generated with 1536 dimensions (Titan v2)
- ✅ Knowledge base returning relevant results
- ✅ Similarity scores above 0.7 for related queries
- ✅ Source attribution working correctly

---

## 🔗 Lab 4: Implement RAG with LangChain

### Step 4.1: Setup LangChain Project Structure

Create the following project structure:

```
bedrock-rag-project/
├── config.py                # Configuration settings
├── rag_pipeline.py          # Main RAG implementation
├── test_rag.py             # Testing script
└── requirements.txt         # Python dependencies
```

```bash
mkdir -p bedrock-rag-project
cd bedrock-rag-project
```

---

### Step 4.2: Create Configuration File

**File:** `config.py`

```python
"""Configuration for Bedrock RAG Pipeline - Updated 2026."""

import os

# AWS Configuration
AWS_REGION = os.environ.get('AWS_REGION', 'us-east-1')
KB_ID = os.environ.get('KB_ID', 'YOUR_KB_ID')  # Replace with your KB ID

# Model Configuration - UPDATED FOR 2026
# See current models: https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html

# Embedding Model
EMBEDDING_MODEL_ID = 'amazon.titan-embed-text-v2:0'  # Updated to v2

# LLM Model
LLM_MODEL_ID = 'anthropic.claude-3-5-sonnet-20240620-v1:0'

# Alternative newer models (check availability in your region):
# LLM_MODEL_ID = 'anthropic.claude-sonnet-4-20250514-v1:0'  # Claude Sonnet 4
# EMBEDDING_MODEL_ID = 'cohere.embed-english-v3'  # Cohere embeddings

# RAG Configuration
DEFAULT_MAX_TOKENS = 4096  # Increased for Claude 3.5+ (supports up to 200k)
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9
DEFAULT_TOP_K = 5  # Number of documents to retrieve

# Chunk Configuration (for analysis)
CHUNK_SIZES = [256, 512, 768, 1024]
CHUNK_OVERLAP = 50

# OpenSearch Configuration (if using direct connection)
OPENSEARCH_ENDPOINT = os.environ.get('OPENSEARCH_ENDPOINT', '')
OPENSEARCH_INDEX = 'bedrock-kb-index'

# Display configuration on import
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Bedrock RAG Pipeline Configuration")
    print("=" * 70)
    print(f"AWS Region: {AWS_REGION}")
    print(f"Knowledge Base ID: {KB_ID}")
    print(f"Embedding Model: {EMBEDDING_MODEL_ID}")
    print(f"LLM Model: {LLM_MODEL_ID}")
    print(f"Max Tokens: {DEFAULT_MAX_TOKENS}")
    print(f"Top-K Retrieval: {DEFAULT_TOP_K}")
    print("=" * 70)
```

---

### Step 4.3: Implement RAG Pipeline

**File:** `rag_pipeline.py`

```python
#!/usr/bin/env python3
"""
Bedrock RAG Pipeline using LangChain
Updated for 2026 - Uses ChatBedrockConverse
"""

import boto3
import json
from typing import List, Dict, Optional
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_aws import ChatBedrockConverse  # UPDATED for 2026
from langchain_aws import AmazonKnowledgeBasesRetriever  # UPDATED package
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
        """
        Setup the Bedrock LLM using ChatBedrockConverse.
        
        UPDATED 2026: Using ChatBedrockConverse instead of BedrockLLM
        for better compatibility with the Bedrock Converse API.
        """
        self.llm = ChatBedrockConverse(
            model=self.model_id,
            region_name=self.region_name,
            model_kwargs={
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "top_p": config.DEFAULT_TOP_P
            }
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
                "success": False,
                "error": str(e)
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
    
    print("\n" + "=" * 70)
    print("Initializing Bedrock RAG Pipeline (2026)")
    print("=" * 70)
    print(f"Model: {config.LLM_MODEL_ID}")
    print(f"Embeddings: {config.EMBEDDING_MODEL_ID}")
    print(f"Knowledge Base: {config.KB_ID}")
    print("=" * 70)
    
    # Initialize pipeline
    rag = BedrockRAGPipeline()
    
    # Example queries
    questions = [
        "What is AWS Lambda and how does it work?",
        "Explain the key features of Amazon Bedrock.",
        "What are the components of RAG architecture?",
        "What is the difference between fixed-size and semantic chunking?"
    ]
    
    print("\nRunning example queries...\n")
    
    for i, question in enumerate(questions, 1):
        print("=" * 70)
        print(f"\nQuestion {i}: {question}")
        print("-" * 70)
        
        # Query the pipeline
        result = rag.query(question)
        
        if result["success"]:
            print(f"\n📝 Answer:\n{result['answer']}\n")
            print(f"📚 Sources ({len(result['source_documents'])} documents):")
            for j, doc in enumerate(result['source_documents'], 1):
                print(f"\n  Source {j}:")
                print(f"  {doc['content'][:150]}...")
                if 'location' in doc['metadata']:
                    print(f"  Location: {doc['metadata']['location']}")
        else:
            print(f"\n❌ Error: {result['answer']}")
        
        print("\n" + "=" * 70)
    
    print("\n✅ All example queries completed!")

if __name__ == "__main__":
    main()
```

> **💡 Important Update:** This tutorial now uses `ChatBedrockConverse` instead of `BedrockLLM`.  
> ChatBedrockConverse uses the newer Bedrock Converse API which provides:
> - Standardized interface across all Bedrock models
> - Better streaming support
> - Improved error handling
> - Support for more models and features
> 
> If you need to use custom/fine-tuned models not supported by Converse API, use `BedrockLLM` from `langchain_community.llms` instead.

---

### Step 4.4: Create Test Script

**File:** `test_rag.py`

```python
#!/usr/bin/env python3
"""Test script for RAG pipeline functionality - Updated 2026."""

import sys
from rag_pipeline import BedrockRAGPipeline
import config

def test_retrieval_only():
    """Test document retrieval without generation."""
    print("\n" + "=" * 70)
    print("TEST 1: Document Retrieval Only")
    print("=" * 70)
    
    rag = BedrockRAGPipeline()
    question = "What is Amazon Bedrock?"
    
    print(f"\nQuery: {question}\n")
    docs = rag.retrieve_only(question)
    
    print(f"Retrieved {len(docs)} documents:\n")
    for i, doc in enumerate(docs, 1):
        print(f"Document {i}:")
        print(f"  Content: {doc['content'][:200]}...")
        print(f"  Metadata: {doc['metadata']}")
        print()

def test_full_rag():
    """Test full RAG pipeline with generation."""
    print("\n" + "=" * 70)
    print("TEST 2: Full RAG Pipeline (Retrieval + Generation)")
    print("=" * 70)
    
    rag = BedrockRAGPipeline()
    question = "What are the main components of a RAG architecture?"
    
    print(f"\nQuery: {question}\n")
    result = rag.query(question)
    
    if result["success"]:
        print("📝 Answer:")
        print(result["answer"])
        print(f"\n✅ Used {len(result['source_documents'])} source documents")
    else:
        print(f"❌ Error: {result['answer']}")

def test_multiple_queries():
    """Test multiple queries to evaluate consistency."""
    print("\n" + "=" * 70)
    print("TEST 3: Multiple Query Test")
    print("=" * 70)
    
    rag = BedrockRAGPipeline()
    
    queries = [
        "What is serverless computing?",
        "How does AWS pricing work?",
        "What are the benefits of using Knowledge Bases in Bedrock?"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\nQuery {i}: {query}")
        print("-" * 70)
        result = rag.query(query)
        if result["success"]:
            print(f"📝 Answer: {result['answer'][:300]}...")
            print(f"✅ Success with {len(result['source_documents'])} sources")
        else:
            print(f"❌ Error: {result['answer']}")
        print()

def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n" + "=" * 70)
    print("TEST 4: Edge Cases")
    print("=" * 70)
    
    rag = BedrockRAGPipeline()
    
    edge_cases = [
        ("", "Empty query"),
        ("x" * 1000, "Very long query"),
        ("What is the meaning of life?", "Out of domain query"),
    ]
    
    for query, description in edge_cases:
        print(f"\n{description}:")
        print(f"  Query: {query[:50]}{'...' if len(query) > 50 else ''}")
        result = rag.query(query)
        print(f"  Success: {result['success']}")
        if result["success"]:
            print(f"  Answer length: {len(result['answer'])} characters")
        else:
            print(f"  Error: {result.get('error', 'Unknown error')}")

def test_model_access():
    """Test if models are accessible before running main tests."""
    import boto3
    
    print("\n" + "=" * 70)
    print("TEST 0: Model Access Verification")
    print("=" * 70)
    
    bedrock_runtime = boto3.client('bedrock-runtime', region_name=config.AWS_REGION)
    
    # Test Claude access
    print("\n🧪 Testing Claude 3.5 Sonnet...")
    try:
        response = bedrock_runtime.invoke_model(
            modelId=config.LLM_MODEL_ID,
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "Hi"}]
            }),
            contentType='application/json',
            accept='application/json'
        )
        print("✅ Claude 3.5 Sonnet: Accessible")
        return True
    except Exception as e:
        if "use case" in str(e).lower():
            print("❌ Claude: Use case form not submitted")
            print("   → Go to Bedrock Console → Playgrounds → Text")
            print("   → Select Claude model and submit the form")
        else:
            print(f"❌ Claude: {str(e)}")
        return False

def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("BEDROCK RAG PIPELINE TEST SUITE (2026)")
    print("=" * 70)
    print(f"\nKnowledge Base ID: {config.KB_ID}")
    print(f"Region: {config.AWS_REGION}")
    print(f"Model: {config.LLM_MODEL_ID}")
    print(f"Embeddings: {config.EMBEDDING_MODEL_ID}")
    
    try:
        # Test model access first
        if not test_model_access():
            print("\n⚠️  Model access test failed. Please fix before proceeding.")
            return
        
        # Run all tests
        test_retrieval_only()
        test_full_rag()
        test_multiple_queries()
        test_edge_cases()
        
        print("\n" + "=" * 70)
        print("✅ ALL TESTS COMPLETED")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    import json  # Import for model access test
    main()
```

---

### Step 4.5: Run the RAG Pipeline

**Set Environment Variables:**

```bash
# Set your Knowledge Base ID
export KB_ID="your-knowledge-base-id"  # Replace with actual KB ID
export AWS_REGION="us-east-1"

# Verify configuration
python config.py
```

**Run the main pipeline:**

```bash
python rag_pipeline.py
```

**Run comprehensive tests:**

```bash
python test_rag.py
```

**Expected Output:**

```
======================================================================
BEDROCK RAG PIPELINE TEST SUITE (2026)
======================================================================

Knowledge Base ID: ABC123XYZ
Region: us-east-1
Model: anthropic.claude-3-5-sonnet-20240620-v1:0
Embeddings: amazon.titan-embed-text-v2:0

======================================================================
TEST 0: Model Access Verification
======================================================================

🧪 Testing Claude 3.5 Sonnet...
✅ Claude 3.5 Sonnet: Accessible

======================================================================
TEST 1: Document Retrieval Only
======================================================================

Query: What is Amazon Bedrock?

Retrieved 5 documents:

Document 1:
  Content: Amazon Bedrock Overview

Amazon Bedrock is a fully managed service that offers a choice of high-performing foundation models (FMs) from leading AI companies like AI21 Labs, Anthropic...
  Metadata: {'location': {'s3Location': {'uri': 's3://bedrock-kb-docs-20260208/documents/bedrock-intro.txt'}}, 'score': 0.8912}

...

======================================================================
TEST 2: Full RAG Pipeline (Retrieval + Generation)
======================================================================

Query: What are the main components of a RAG architecture?

📝 Answer:
Based on the provided context, RAG (Retrieval-Augmented Generation) architecture consists of four main components:

1. **Document Processing Pipeline**: This handles document ingestion from various sources, text extraction to convert documents to plain text, chunking to break text into manageable pieces, and embedding generation to convert text chunks to vector representations.

2. **Vector Database**: This stores document embeddings and enables semantic search. It returns the most relevant chunks based on query similarity. Examples include OpenSearch, Pinecone, Chroma, FAISS, and S3 Vectors.

3. **Retrieval Mechanism**: This component handles query embedding (converting user questions to vectors), similarity search (finding nearest neighbors in vector space), ranking and ordering results by relevance, optional reranking for improved precision, and context selection (choosing top-k results for generation).

4. **Generation Component**: This handles prompt construction by building context with retrieved information, LLM invocation to generate responses using the context, response formatting to present answers to users, and citation tracking to attribute sources.

✅ Used 2 source documents

======================================================================
TEST 3: Multiple Query Test
======================================================================

Query 1: What is serverless computing?
----------------------------------------------------------------------
📝 Answer: Serverless computing is a cloud computing execution model where the cloud provider dynamically manages the allocation and provisioning of servers. In the context provided, AWS Lambda is mentioned as a serverless computing platform, which is one of the key AWS services...
✅ Success with 3 sources

...

======================================================================
✅ ALL TESTS COMPLETED
======================================================================
```

---

**✅ Lab 4 Checkpoint - You should see:**

- ✅ RAG pipeline initialized successfully
- ✅ Documents successfully retrieved from Knowledge Base
- ✅ Claude 3.5 Sonnet generating contextual answers
- ✅ Source citations included in responses
- ✅ All tests passing without errors
- ✅ Relevant answers to all test queries

---

## ⚡ Lab 5: Test and Optimize Performance

### Step 5.1: Create Performance Testing Script

**File:** `test_performance.py`

```python
#!/usr/bin/env python3
"""
Test and analyze RAG pipeline performance.
Focus on retrieval quality and latency.
"""

import boto3
import json
import time
from typing import List, Dict, Tuple
from datetime import datetime
import config

class PerformanceTester:
    """Test retrieval performance and analyze results."""
    
    def __init__(self, kb_id: str, region: str = 'us-east-1'):
        self.kb_id = kb_id
        self.region = region
        self.bedrock_agent_runtime = boto3.client(
            service_name='bedrock-agent-runtime',
            region_name=region
        )
    
    def retrieve_with_timing(
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
            print(f"❌ Error during retrieval: {str(e)}")
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
        print("\n" + "=" * 70)
        print("RAG PERFORMANCE TEST SUITE")
        print("=" * 70)
        print(f"Knowledge Base: {self.kb_id}")
        print(f"Test Queries: {len(test_queries)}")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
        
        all_results = []
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n📝 Query {i}/{len(test_queries)}: {query}")
            print("-" * 70)
            
            results, latency = self.retrieve_with_timing(query, num_results=5)
            metrics = self.evaluate_results(results, query)
            
            print(f"⏱️  Latency: {latency:.2f} ms")
            print(f"📊 Results Retrieved: {metrics['num_results']}")
            print(f"🎯 Average Relevance Score: {metrics['avg_score']:.4f}")
            print(f"📏 Average Chunk Length: {metrics['avg_length']} characters")
            
            # Display top result
            if results:
                print(f"\n✅ Top Result (Score: {results[0]['score']:.4f}):")
                print(f"   {results[0]['content']['text'][:150]}...")
            
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
        
        print("\n" + "=" * 70)
        print("AGGREGATE STATISTICS")
        print("=" * 70)
        print(f"Average Latency: {avg_latency:.2f} ms")
        print(f"Average Relevance Score: {avg_score:.4f}")
        print(f"Average Chunk Length: {avg_length} characters")
        
        # Performance assessment
        print("\n" + "=" * 70)
        print("PERFORMANCE ASSESSMENT")
        print("=" * 70)
        
        if avg_latency < 500:
            print("✅ Latency: Excellent (<500ms)")
        elif avg_latency < 1000:
            print("⚠️  Latency: Good (500-1000ms)")
        else:
            print("❌ Latency: Needs improvement (>1000ms)")
        
        if avg_score > 0.7:
            print("✅ Relevance: Excellent (>0.7)")
        elif avg_score > 0.5:
            print("⚠️  Relevance: Good (0.5-0.7)")
        else:
            print("❌ Relevance: Needs improvement (<0.5)")
        
        print("=" * 70)
        
        return {
            'individual_results': all_results,
            'aggregate': {
                'avg_latency_ms': avg_latency,
                'avg_relevance_score': avg_score,
                'avg_chunk_length': avg_length
            }
        }

def main():
    """Main execution function."""
    
    # Comprehensive test queries covering different complexities
    test_queries = [
        # Short, specific queries
        "What is AWS Lambda?",
        "What is Amazon Bedrock?",
        
        # Medium complexity queries
        "Explain the components of RAG architecture",
        "How does AWS pricing work?",
        
        # Complex, multi-part queries
        "What are the differences between fixed-size chunking and semantic chunking?",
        "Describe the complete workflow of creating a Knowledge Base in Amazon Bedrock",
        
        # Domain-specific queries
        "What are the best practices for RAG implementation?",
        "How do I optimize retrieval performance in a knowledge base?"
    ]
    
    # Initialize tester
    tester = PerformanceTester(
        kb_id=config.KB_ID,
        region=config.AWS_REGION
    )
    
    # Run tests
    results = tester.run_test_suite(test_queries)
    
    # Save results to file
    output_file = f"performance_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    print("\n" + "=" * 70)
    print("✅ Performance testing completed!")
    print("=" * 70)

if __name__ == "__main__":
    main()
```

**Run performance tests:**

```bash
python test_performance.py
```

**Expected Output:**

```
======================================================================
RAG PERFORMANCE TEST SUITE
======================================================================
Knowledge Base: ABC123XYZ
Test Queries: 8
Timestamp: 2026-02-08 14:30:45
======================================================================

📝 Query 1/8: What is AWS Lambda?
----------------------------------------------------------------------
⏱️  Latency: 245.67 ms
📊 Results Retrieved: 5
🎯 Average Relevance Score: 0.8123
📏 Average Chunk Length: 456 characters

✅ Top Result (Score: 0.8456):
   Lambda: Serverless computing platform...

...

======================================================================
AGGREGATE STATISTICS
======================================================================
Average Latency: 287.34 ms
Average Relevance Score: 0.7845
Average Chunk Length: 423 characters

======================================================================
PERFORMANCE ASSESSMENT
======================================================================
✅ Latency: Excellent (<500ms)
✅ Relevance: Excellent (>0.7)
======================================================================

💾 Results saved to: performance_results_20260208_143045.json

======================================================================
✅ Performance testing completed!
======================================================================
```

---

### Step 5.2: Chunking Strategy Analysis

Understanding the chunking strategy used by your Knowledge Base:

**File:** `analyze_chunking.py`

```python
#!/usr/bin/env python3
"""
Analyze chunking strategy and provide recommendations.
"""

import json

def analyze_chunking_strategy():
    """Provide information about chunking strategies."""
    
    print("\n" + "=" * 70)
    print("CHUNKING STRATEGY ANALYSIS")
    print("=" * 70)
    
    print("""
Current Knowledge Base Configuration:

When you created your Knowledge Base, AWS Bedrock automatically configured
the chunking strategy. The default settings are:

📊 Default Chunking Parameters:
   - Chunk Size: ~300 tokens (~400-500 characters)
   - Overlap: ~20% between chunks
   - Strategy: Semantic chunking with sentence boundary detection
   - Model: Uses foundation models for intelligent splitting

🆕 Advanced Chunking Options (2026):
   1. Fixed-Size Chunking
      - Uniform chunk sizes
      - Predictable token counts
      - Best for: Structured documents, API docs

   2. Semantic Chunking
      - Respects semantic boundaries
      - Preserves context
      - Best for: Narrative content, articles

   3. Hierarchical Chunking
      - Parent-child relationships
      - Multi-level retrieval
      - Best for: Books, technical manuals

   4. Custom Chunking
      - Lambda function preprocessing
      - Complete control
      - Best for: Special formats, domain-specific needs

📈 Optimization Recommendations:

For Better Precision (Specific Answers):
   → Use smaller chunks (256-400 tokens)
   → Increase overlap (25-30%)
   → Enable reranking

For Better Context (Comprehensive Answers):
   → Use larger chunks (600-1024 tokens)
   → Standard overlap (15-20%)
   → Retrieve more documents (top-k = 7-10)

For Mixed Content:
   → Use semantic or hierarchical chunking
   → Test different configurations
   → Monitor relevance scores

🔬 Testing Different Chunk Sizes:

To test different chunk sizes, you would need to:
1. Create multiple Knowledge Bases with different configurations
2. OR manually process documents and upload pre-chunked content
3. OR use custom data sources with different preprocessing

For this lab, we're using the default configuration which works well
for most use cases.

💡 Next Steps:
   - Monitor performance metrics from test_performance.py
   - If avg_score < 0.7, consider adjusting chunk size
   - If latency > 500ms, consider reducing top-k or using caching
    """)
    
    print("=" * 70)

if __name__ == "__main__":
    analyze_chunking_strategy()
```

**Run chunking analysis:**

```bash
python analyze_chunking.py
```

---

**✅ Lab 5 Checkpoint - You should see:**

- ✅ Performance metrics for each query
- ✅ Latency measurements (< 500ms ideal)
- ✅ Relevance score analysis (> 0.7 is excellent)
- ✅ Aggregate statistics calculated
- ✅ Performance assessment with recommendations
- ✅ Understanding of chunking strategies

---

## 🤖 Lab 6: Advanced Agent Integration

### Step 6.1: Understanding Agent Integration

AgentCore allows you to create autonomous agents that can:
- Use tools and APIs
- Make decisions based on context
- Execute multi-step workflows
- Integrate with your RAG pipeline

### Step 6.2: Create Advanced Agent with RAG

**File:** `agent_with_rag.py`

```python
#!/usr/bin/env python3
"""
Advanced Agent integration with RAG pipeline for complex workflows.
Updated for 2026 - Uses ChatBedrockConverse
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
            'query': query,
            'timestamp': datetime.now().isoformat()
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
        sources = []
        if needs_rag and use_rag:
            rag_result = self.rag_pipeline.query(query)
            if rag_result['success']:
                context = rag_result.get('answer', '')
                sources = rag_result.get('source_documents', [])
                
                reasoning_steps.append({
                    'step': 'rag_retrieval',
                    'description': 'Retrieved relevant documents from knowledge base',
                    'num_sources': len(sources),
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
            'used_rag': needs_rag and use_rag,
            'sources': len(sources)
        })
        
        return {
            'response': response,
            'reasoning_steps': reasoning_steps,
            'used_rag': needs_rag and use_rag,
            'context': context,
            'sources': sources
        }
    
    def _needs_knowledge_base(self, query: str) -> bool:
        """
        Determine if query requires knowledge base retrieval.
        
        Uses keyword-based heuristic. In production, use more
        sophisticated classification.
        """
        knowledge_keywords = [
            'what is', 'what are', 'how does', 'explain',
            'describe', 'tell me about', 'definition',
            'features', 'components', 'architecture',
            'details about', 'information on'
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
        print("\n" + "=" * 70)
        print(f"MULTI-STEP WORKFLOW: {task}")
        print("=" * 70)
        
        workflow_steps = []
        
        # Step 1: Break down task
        print("\n🔍 Step 1: Breaking down task...")
        subtasks = self._decompose_task(task)
        workflow_steps.append({
            'step': 'task_decomposition',
            'subtasks': subtasks
        })
        print(f"   Identified {len(subtasks)} subtasks")
        
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
            print(f"   ✅ Completed with {len(result.get('sources', []))} sources")
        
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
            'final_answer': final_answer,
            'num_subtasks': len(subtasks)
        }
    
    def _decompose_task(self, task: str) -> List[str]:
        """Break down complex task into subtasks."""
        # Simple decomposition - in production, use LLM for this
        task_lower = task.lower()
        
        if "summarize" in task_lower and "compare" in task_lower:
            return [
                "What are the main AWS services?",
                "How does AWS pricing work?",
                "Compare pricing models across services"
            ]
        elif "explain" in task_lower and "architecture" in task_lower:
            return [
                f"What is {task.replace('explain', '').replace('architecture', '').strip()}?",
                "What are the key components?",
                "What are the use cases?"
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
        combined_info = "\n\n".join([
            f"Subtask: {r.get('reasoning_steps', [{}])[0].get('query', '')}\n"
            f"Answer: {r['response']}"
            for r in results
        ])
        
        synthesis_prompt = f"""Based on the following information, provide a comprehensive answer to this task:

Task: {original_task}

Information gathered:
{combined_info}

Comprehensive Answer (synthesize the information above):"""
        
        return self._generate_response(original_task, synthesis_prompt)
    
    def get_conversation_summary(self) -> Dict:
        """Get summary of conversation history."""
        return {
            'total_exchanges': len(self.conversation_history),
            'rag_usage_count': sum(1 for h in self.conversation_history if h['used_rag']),
            'total_sources_used': sum(h['sources'] for h in self.conversation_history),
            'recent_queries': [h['query'] for h in self.conversation_history[-5:]]
        }

def main():
    """Example usage of RAG Agent."""
    
    print("\n" + "=" * 70)
    print("BEDROCK RAG AGENT WITH ADVANCED INTEGRATION (2026)")
    print("=" * 70)
    print(f"Model: {config.LLM_MODEL_ID}")
    print(f"Knowledge Base: {config.KB_ID}")
    print("=" * 70)
    
    # Initialize agent
    agent = RAGAgent()
    
    # Example 1: Simple query with reasoning
    print("\n\n📝 Example 1: Simple Query with Reasoning")
    print("=" * 70)
    
    result = agent.process_with_reasoning(
        "What is Amazon Bedrock?"
    )
    
    print(f"\n💬 Response: {result['response'][:200]}...")
    print(f"\n🔍 Used RAG: {result['used_rag']}")
    print(f"📚 Sources: {len(result.get('sources', []))}")
    print("\n🧠 Reasoning Steps:")
    for step in result['reasoning_steps']:
        print(f"   → {step['description']}")
    
    # Example 2: Multi-step workflow
    print("\n\n📝 Example 2: Multi-Step Workflow")
    print("=" * 70)
    
    workflow_result = agent.multi_step_workflow(
        "Explain RAG architecture and its main components"
    )
    
    print(f"\n\n📖 Final Answer:")
    print(workflow_result['final_answer'][:400] + "...")
    
    # Example 3: Conversation history
    print("\n\n📝 Example 3: Conversational Flow")
    print("=" * 70)
    
    queries = [
        "What is AWS Lambda?",
        "How does it differ from traditional servers?",
        "What are the pricing models?"
    ]
    
    for query in queries:
        result = agent.process_with_reasoning(query)
        print(f"\nQ: {query}")
        print(f"A: {result['response'][:150]}...")
    
    # Display conversation summary
    summary = agent.get_conversation_summary()
    print("\n\n📊 Conversation Summary")
    print("=" * 70)
    print(f"Total Exchanges: {summary['total_exchanges']}")
    print(f"RAG Usage: {summary['rag_usage_count']} times")
    print(f"Total Sources Used: {summary['total_sources_used']}")
    print("\nRecent Queries:")
    for q in summary['recent_queries']:
        print(f"  • {q}")
    
    print("\n" + "=" * 70)
    print("✅ All examples completed!")
    print("=" * 70)

if __name__ == "__main__":
    main()
```

**Run the agent:**

```bash
python agent_with_rag.py
```

**Expected Output:**

```
======================================================================
BEDROCK RAG AGENT WITH ADVANCED INTEGRATION (2026)
======================================================================
Model: anthropic.claude-3-5-sonnet-20240620-v1:0
Knowledge Base: ABC123XYZ
======================================================================


📝 Example 1: Simple Query with Reasoning
======================================================================

💬 Response: Amazon Bedrock is a fully managed service that offers a choice of high-performing foundation models from leading AI companies. It provides access to models from Anthropic, Amazon, AI21 Labs, Cohere...

🔍 Used RAG: True
📚 Sources: 2

🧠 Reasoning Steps:
   → Analyzing user query to determine intent
   → Determining if knowledge base retrieval is needed
   → Retrieved relevant documents from knowledge base
   → Generated final response


📝 Example 2: Multi-Step Workflow
======================================================================

======================================================================
MULTI-STEP WORKFLOW: Explain RAG architecture and its main components
======================================================================

🔍 Step 1: Breaking down task...
   Identified 3 subtasks

⚙️  Step 2: Executing subtask - What is RAG architecture?
   ✅ Completed with 2 sources

⚙️  Step 3: Executing subtask - What are the key components?
   ✅ Completed with 3 sources

⚙️  Step 4: Executing subtask - What are the use cases?
   ✅ Completed with 2 sources

🔄 Final Step: Synthesizing results...

✅ Workflow completed!


📖 Final Answer:
RAG (Retrieval-Augmented Generation) architecture is a technique that enhances Large Language Models by combining information retrieval with language generation. Instead of relying solely on the model's training data, RAG dynamically retrieves relevant information from external knowledge sources to generate more accurate responses.

The RAG architecture consists of four main components:

1. Document Processing Pipeline: This handles document ingestion from various sources, text extraction...


📝 Example 3: Conversational Flow
======================================================================

Q: What is AWS Lambda?
A: AWS Lambda is a serverless computing platform offered by Amazon Web Services. It allows you to run code without provisioning or managing servers...

Q: How does it differ from traditional servers?
A: Unlike traditional servers where you manage the infrastructure, Lambda is serverless, meaning AWS handles all the infrastructure management...

Q: What are the pricing models?
A: AWS Lambda follows a pay-as-you-go pricing model. You only pay for the compute time you consume...


📊 Conversation Summary
======================================================================
Total Exchanges: 7
RAG Usage: 7 times
Total Sources Used: 17

Recent Queries:
  • What is Amazon Bedrock?
  • What is RAG architecture?
  • What are the key components?
  • What are the use cases?
  • What is AWS Lambda?

======================================================================
✅ All examples completed!
======================================================================
```

---

**✅ Lab 6 Checkpoint - You should see:**

- ✅ Agent processing queries with reasoning chains
- ✅ RAG integration working seamlessly
- ✅ Multi-step workflows executing successfully
- ✅ Conversation history tracking functional
- ✅ Task decomposition and synthesis working
- ✅ Source attribution across all responses

---

## 🔧 Troubleshooting

### Common Issues and Solutions

#### Issue 1: "Use Case Form Required" Error (Anthropic Models)

**Symptoms:**
```
ValidationException: Before you can use Anthropic models, you must submit 
use case details for your account.
```

**Solution:**

**Option 1 - Console (Easiest):**
1. Go to: Bedrock Console → Playgrounds → Text
2. Click "Select model" → Choose any Claude model
3. Use case form appears automatically
4. Fill in your use case description
5. Click Submit
6. ✅ Access granted immediately

**Option 2 - AWS CLI:**
```bash
aws bedrock put-use-case-for-model-access \
  --model-id anthropic.claude-3-5-sonnet-20240620-v1:0 \
  --use-case "Internal document Q&A system" \
  --region us-east-1
```

**Note:** This is a one-time requirement per AWS account. Once submitted, all Anthropic models become immediately available.

---

#### Issue 2: Knowledge Base Not Returning Results

**Symptoms:**
- Empty results from queries
- Low relevance scores
- "No documents found" errors

**Solutions:**

**1. Check Data Source Sync:**
- Go to: Bedrock Console → Knowledge bases → Your KB → Data source tab
- Verify "Sync status" is **"Completed"**
- Check "Documents synced" count (should be > 0)
- If needed, click **"Sync"** to re-sync

**2. Verify Documents in S3:**
- Go to: S3 Console → Your bucket → `documents` folder
- Confirm files are present and not empty
- Check file permissions

**3. Re-sync if Necessary:**
- Select your data source
- Click **"Sync"**
- Wait 5-10 minutes for completion

---

#### Issue 3: "Access Denied" Errors

**Symptoms:**
- 403 Forbidden errors
- "Not authorized to perform action"

**Solutions:**

**1. Check IAM Role Permissions:**
- Go to: IAM Console → Roles → BedrockKnowledgeBaseRole
- Verify policy includes:
  - S3 read access to your bucket
  - Bedrock InvokeModel permission
  - OpenSearch API access

**2. Check Data Access Policy:**
- Go to: OpenSearch Console → Serverless → Security → Data access policies
- Verify `bedrock-kb-data-access-policy` includes your role

**3. Verify Model Access:**
- Models should be automatically accessible
- For Anthropic: Ensure use case form submitted
- Test with: `python test_model_access.py`

---

#### Issue 4: Python Package Import Errors

**Symptoms:**
```
ModuleNotFoundError: No module named 'langchain_aws'
ImportError: cannot import name 'ChatBedrockConverse'
```

**Solutions:**

```bash
# Reinstall packages with correct versions
pip install --upgrade pip
pip install --upgrade \
  boto3 \
  langchain>=0.3.0 \
  langchain-aws>=1.2.2 \
  langchain-community>=0.3.0 \
  opensearch-py

# Verify installation
pip list | grep langchain

# Check specific package versions
pip show langchain-aws  # Should be 1.2.2+
pip show langchain      # Should be 0.3.0+

# If using virtual environment, ensure it's activated
source bedrock-rag-env/bin/activate  # Unix/Mac
.\bedrock-rag-env\Scripts\Activate.ps1  # Windows
```

---

#### Issue 5: Vector Field Name Mismatch

**Symptoms:**
```
Error: Field 'bedrock-kb-vector' is not knn_vector type
ValidationException: Vector field not found
```

**Solution:**

The field names must be **EXACTLY** as specified:

```
Vector field:   bedrock-knowledge-base-default-vector
Text field:     AMAZON_BEDROCK_TEXT_CHUNK
Metadata field: AMAZON_BEDROCK_METADATA
```

**To Fix:**
1. Delete the Knowledge Base
2. Re-run the vector index script to verify field names
3. When creating KB in console, **copy-paste** field names (don't type)
4. Triple-check spelling and capitalization

---

#### Issue 6: Slow Retrieval Performance

**Symptoms:**
- Queries taking >2 seconds
- Timeouts
- High latency

**Solutions:**

**1. Reduce Retrieved Documents:**
```python
# In your code
retrieval_config={
    "vectorSearchConfiguration": {
        "numberOfResults": 3  # Reduced from 5
    }
}
```

**2. Check OpenSearch Collection Health:**
- Go to: OpenSearch Console → Serverless → Collections
- Verify `bedrock-kb-collection` status is **Active**
- Check for any error messages

**3. Consider Caching:**
- Implement caching for frequent queries
- Use in-memory cache (Redis) for production
- Cache embedding generation results

**4. Optimize Network:**
- Ensure you're in the same region as your KB
- Check your network connectivity
- Consider using VPC endpoints for production

---

#### Issue 7: OpenSearch Collection Creation Failed

**Symptoms:**
- Collection stuck in "Creating" state
- Error during collection creation

**Solutions:**

**1. Verify Security Policies Exist FIRST:**
- Encryption policy: `bedrock-kb-encryption-policy` ✅
- Network policy: `bedrock-kb-network-policy` ✅
- These MUST exist before creating collection

**2. Check Policy Configuration:**
- Policy rules must match collection name pattern exactly
- Resource type must be correct
- Collection name pattern: `bedrock-kb-collection`

**3. Recreate if Necessary:**
- Delete failed collection (if it exists)
- Verify both policies are "Active"
- Wait 1-2 minutes
- Recreate collection

---

### Enable Debug Logging

For detailed debugging, add this to your Python scripts:

```python
import logging

# Enable debug logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Reduce boto3 logging noise (optional)
logging.getLogger('boto3').setLevel(logging.WARNING)
logging.getLogger('botocore').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
```

---

### Getting Help

If issues persist:

**1. AWS Service Health Dashboard:**
- Check: https://health.aws.amazon.com/health/status
- Look for Bedrock or OpenSearch service issues

**2. CloudWatch Logs:**
- Go to: CloudWatch → Log groups
- Look for Bedrock and OpenSearch logs
- Search for error messages around the time of your issue

**3. AWS Support:**
- File a support case if you have a support plan
- Include: error messages, timestamps, KB ID, region

**4. AWS re:Post Community:**
- Ask questions: https://repost.aws/
- Tag with: amazon-bedrock, opensearch

**5. Check AWS Documentation:**
- Bedrock Docs: https://docs.aws.amazon.com/bedrock/
- OpenSearch Docs: https://docs.aws.amazon.com/opensearch-service/
- Model Lifecycle: https://docs.aws.amazon.com/bedrock/latest/userguide/model-lifecycle.html

---

## 🧹 Cleanup

> **⚠️ IMPORTANT:** To avoid unexpected charges, clean up all resources after completing the lab.

### Cleanup Checklist

Follow these steps in order:

- [ ] Delete Knowledge Base
- [ ] Delete OpenSearch Serverless Collection
- [ ] Delete OpenSearch Policies (3 policies)
- [ ] Empty and Delete S3 Bucket
- [ ] Delete IAM Role
- [ ] Delete Local Files (optional)

### Step 1: Delete Knowledge Base

1. **Go to Bedrock Console:**
   - https://console.aws.amazon.com/bedrock
   - Click **"Knowledge bases"**

2. **Delete Knowledge Base:**
   - Select your knowledge base (`my-rag-knowledge-base`)
   - Click **"Delete"**
   - Type the knowledge base name to confirm
   - Click **"Delete"**
   - Wait for deletion to complete (~1 minute)

---

### Step 2: Delete OpenSearch Serverless Collection

1. **Go to OpenSearch Console:**
   - https://console.aws.amazon.com/aos
   - Click **"Serverless"** → **"Collections"**

2. **Delete Collection:**
   - Select `bedrock-kb-collection`
   - Click **"Delete"**
   - Type `delete` to confirm
   - Click **"Delete"**
   - Wait for deletion (~2-3 minutes)

---

### Step 3: Delete OpenSearch Policies

Delete in this order:

**1. Delete Data Access Policy:**
- Go to: Serverless → Security → Data access policies
- Select `bedrock-kb-data-access-policy`
- Click **"Delete"**
- Confirm deletion

**2. Delete Network Policy:**
- Go to: Serverless → Security → Network policies
- Select `bedrock-kb-network-policy`
- Click **"Delete"**
- Confirm deletion

**3. Delete Encryption Policy:**
- Go to: Serverless → Security → Encryption policies
- Select `bedrock-kb-encryption-policy`
- Click **"Delete"**
- Confirm deletion

---

### Step 4: Empty and Delete S3 Bucket

1. **Go to S3 Console:**
   - https://console.aws.amazon.com/s3

2. **Empty Bucket:**
   - Click on your bucket name
   - Click **"Empty"**
   - Type `permanently delete` to confirm
   - Click **"Empty"**
   - Wait for completion

3. **Delete Bucket:**
   - Go back to S3 bucket list
   - Select your bucket
   - Click **"Delete"**
   - Type the bucket name to confirm
   - Click **"Delete bucket"**

---

### Step 5: Delete IAM Role and Policy

**1. Delete IAM Role:**
- Go to: https://console.aws.amazon.com/iam
- Click **"Roles"**
- Search for `BedrockKnowledgeBaseRole`
- Select the role
- Click **"Delete"**
- Type the role name to confirm
- Click **"Delete"**

**2. Delete IAM Policy:**
- Click **"Policies"**
- Search for `BedrockKnowledgeBasePolicy`
- Select the policy
- Click **"Actions"** → **"Delete"**
- Type the policy name to confirm
- Click **"Delete"**

---

### Step 6: Clean Up Local Files (Optional)

```bash
# Remove project directory
cd ..
rm -rf bedrock-rag-project

# Remove sample documents
rm -rf sample-docs

# Remove test results
rm -f performance_results_*.json
rm -f create_vector_index.py
rm -f test_*.py

# Deactivate virtual environment
deactivate

# Remove virtual environment (optional)
rm -rf bedrock-rag-env
```

---

### Final Verification

**✅ Verify all resources are deleted:**

1. **Bedrock Console:**
   - No Knowledge Bases listed

2. **OpenSearch Console:**
   - No Collections
   - No Policies (encryption, network, data access)

3. **S3 Console:**
   - Bucket deleted

4. **IAM Console:**
   - Role and policy deleted

**💰 Cost Verification:**
- Wait 24-48 hours
- Check AWS Cost Explorer
- Verify no ongoing charges from this lab

---

## 📚 Additional Resources

### AWS Documentation

**Bedrock:**
- [Amazon Bedrock User Guide](https://docs.aws.amazon.com/bedrock/latest/userguide/)
- [Bedrock Knowledge Bases](https://docs.aws.amazon.com/bedrock/latest/userguide/knowledge-base.html)
- [Bedrock API Reference](https://docs.aws.amazon.com/bedrock/latest/APIReference/)
- [Model Lifecycle](https://docs.aws.amazon.com/bedrock/latest/userguide/model-lifecycle.html)
- [Model Access Guide (2025 Updates)](https://aws.amazon.com/blogs/security/simplified-amazon-bedrock-model-access/)

**OpenSearch:**
- [OpenSearch Serverless Guide](https://docs.aws.amazon.com/opensearch-service/latest/developerguide/serverless.html)
- [Vector Search](https://opensearch.org/docs/latest/search-plugins/knn/index/)

**S3:**
- [S3 User Guide](https://docs.aws.amazon.com/s3/index.html)
- [S3 Vectors (Preview)](https://docs.aws.amazon.com/s3/latest/userguide/s3-vectors.html)

### LangChain Resources

- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [LangChain AWS Integration](https://python.langchain.com/docs/integrations/platforms/aws)
- [ChatBedrockConverse](https://python.langchain.com/docs/integrations/chat/bedrock/)
- [Retrieval QA Chains](https://python.langchain.com/docs/use_cases/question_answering/)
- [LangChain-AWS Package](https://github.com/langchain-ai/langchain-aws)

### RAG and AI Concepts

**Research Papers:**
- [RAG Paper (Lewis et al. 2020)](https://arxiv.org/abs/2005.11401)
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP](https://ai.meta.com/research/publications/retrieval-augmented-generation-for-knowledge-intensive-nlp-tasks/)

**Tutorials:**
- [AWS RAG Blog Posts](https://aws.amazon.com/blogs/machine-learning/tag/retrieval-augmented-generation/)
- [Building Production RAG Systems](https://www.anthropic.com/index/building-effective-agents)

### Best Practices

- [AWS Well-Architected Framework - ML Lens](https://docs.aws.amazon.com/wellarchitected/latest/machine-learning-lens/)
- [Bedrock Best Practices](https://docs.aws.amazon.com/bedrock/latest/userguide/best-practices.html)
- [OpenSearch Best Practices](https://docs.aws.amazon.com/opensearch-service/latest/developerguide/bp.html)

### Community and Support

- [AWS re:Post - Bedrock](https://repost.aws/tags/TA4IHvpzj4TgK_1P_chZdJKA/amazon-bedrock)
- [LangChain Discord](https://discord.gg/langchain)
- [AWS Developer Forums](https://forums.aws.amazon.com/)

### Video Tutorials

- [AWS Bedrock Workshop](https://catalog.workshops.aws/amazon-bedrock/)
- [Generative AI on AWS](https://aws.amazon.com/generative-ai/learning-path/)

---

## 🎓 What You've Learned

Congratulations! You've completed the AWS Bedrock Knowledge Base with RAG implementation lab. You now know how to:

### ✅ Setup & Configuration
- Navigate the simplified 2026 model access process
- Enable Bedrock model access (including Anthropic use case forms)
- Create and configure Knowledge Bases
- Setup OpenSearch Serverless for vector storage
- Configure IAM roles and permissions properly

### ✅ Document Processing
- Upload documents to S3
- Trigger document ingestion and embedding
- Work with Titan Embeddings v2
- Verify successful processing

### ✅ RAG Implementation
- Implement RAG pipeline with LangChain
- Use ChatBedrockConverse (2026 best practice)
- Generate embeddings using Titan v2
- Retrieve relevant documents
- Generate contextual responses with Claude 3.5 Sonnet

### ✅ Performance Optimization
- Test retrieval with different queries
- Measure latency and relevance
- Understand chunking strategies
- Optimize for your use case

### ✅ Advanced Integration
- Create agents with RAG capabilities
- Implement multi-step reasoning
- Build complex workflows
- Track conversation history

---

## 🚀 Next Steps

### Intermediate Projects

1. **Multimodal RAG:**
   - Add image understanding to your knowledge base
   - Use Titan Multimodal Embeddings G1
   - Process documents with charts and diagrams

2. **Custom Chunking:**
   - Implement document-specific chunking strategies
   - Use Lambda functions for preprocessing
   - Test hierarchical chunking

3. **Hybrid Search:**
   - Combine keyword and semantic search
   - Implement BM25 + vector search
   - Add metadata filtering

4. **Caching Layer:**
   - Add Redis for query caching
   - Implement embedding cache
   - Optimize for frequently asked questions

### Advanced Projects

1. **Fine-tuning:**
   - Customize foundation models with your data
   - Implement continued pre-training
   - Evaluate custom models

2. **Guardrails:**
   - Implement content filtering
   - Add safety checks
   - Control model responses

3. **Production Deployment:**
   - Deploy with API Gateway and Lambda
   - Add authentication and authorization
   - Implement rate limiting

4. **Monitoring:**
   - Add CloudWatch metrics and alerting
   - Track usage and costs
   - Monitor performance

### Real-World Applications

- **Customer Support Chatbots:** 24/7 automated support with knowledge base
- **Internal Documentation Search:** Find company information quickly
- **Research Assistant Tools:** Analyze documents and extract insights
- **Code Documentation Q&A:** Help developers find answers in codebases
- **Compliance Assistant:** Answer regulatory questions from policy documents

---

## 📝 Feedback

Found an issue or have suggestions? We'd love to hear from you:

- **GitHub Issues:** Report bugs or request features
- **AWS Support:** For service-specific issues
- **Training Coordinator:** For course-related feedback

---

## 📄 Version History

- **v3.0 (February 2026):** Major update for 2026 features
  - Updated model access process (October 2025 changes)
  - ChatBedrockConverse integration
  - Titan Embeddings v2
  - New multimodal and S3 Vectors features
  - Improved error handling and troubleshooting
  
- **v2.0 (2025):** Initial comprehensive version
- **v1.0 (2024):** Original release

---

## 🏆 Credits

**Created by:** AWS Training Team  
**Last Updated:** February 2026  
**Maintained by:** AWS Bedrock Education Team

**Contributors:**
- AWS Bedrock Product Team
- AWS Solutions Architects
- Community Contributors

---

**License:** This tutorial is provided under the MIT-0 License. See LICENSE file for details.

**Disclaimer:** This tutorial is for educational purposes. Always follow AWS best practices and security guidelines for production deployments.

---
