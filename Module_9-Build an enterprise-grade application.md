# Enterprise E-commerce AI Agent with AWS Bedrock

## Complete Guide: Product Discovery & Customer Support System

**Estimated Completion Time:** 3-4 hours  
**Difficulty Level:** Advanced  
**Last Updated:** February 2026

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Prerequisites](#prerequisites)
4. [Part 1: Environment Setup](#part-1-environment-setup)
5. [Part 2: Knowledge Base with RAG](#part-2-knowledge-base-with-rag)
6. [Part 3: Bedrock Agent Configuration](#part-3-bedrock-agent-configuration)
7. [Part 4: Guardrails Implementation](#part-4-guardrails-implementation)
8. [Part 5: LangChain Integration](#part-5-langchain-integration)
9. [Part 6: Fine-tuning Considerations](#part-6-fine-tuning-considerations)
10. [Part 7: Testing & Validation](#part-7-testing--validation)
11. [Troubleshooting](#troubleshooting)
12. [Cleanup](#cleanup)
13. [Next Steps](#next-steps)

---

## Overview

### What You'll Build

An enterprise-grade E-commerce AI Agent that combines multiple AWS Bedrock capabilities:

- **Product Discovery**: Semantic search across product catalogs using RAG
- **Customer Support**: Intelligent responses using agent workflows
- **Safety Controls**: Guardrails for content filtering and compliance
- **Custom Intelligence**: Fine-tuned model integration (optional)
- **Orchestration**: LangChain for complex workflows

### Key Components

```
┌─────────────────────────────────────────────────────────────┐
│                    E-commerce AI Agent                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐ │
│  │   Customer   │───▶│   Guardrails │───▶│    Agent     │ │
│  │    Query     │    │   (Safety)   │    │    Core      │ │
│  └──────────────┘    └──────────────┘    └──────┬───────┘ │
│                                                   │          │
│                      ┌────────────────────────────┼─────┐   │
│                      │                            ▼     │   │
│                      │  ┌──────────────────────────┐   │   │
│                      │  │   Knowledge Base (RAG)    │   │   │
│                      │  │  - Product Catalog        │   │   │
│                      │  │  - FAQs                   │   │   │
│                      │  │  - Policies               │   │   │
│                      │  └──────────────────────────┘   │   │
│                      │                                  │   │
│                      │  ┌──────────────────────────┐   │   │
│                      │  │   Action Groups           │   │   │
│                      │  │  - Check Inventory        │   │   │
│                      │  │  - Process Order          │   │   │
│                      │  │  - Track Shipping         │   │   │
│                      │  └──────────────────────────┘   │   │
│                      │                                  │   │
│                      │  ┌──────────────────────────┐   │   │
│                      │  │   Foundation Model        │   │   │
│                      │  │  - Claude 3.5 Sonnet     │   │   │
│                      │  │  - Custom Fine-tuned     │   │   │
│                      │  └──────────────────────────┘   │   │
│                      └──────────────────────────────────┘   │
│                                                              │
│                      ┌──────────────────────────┐           │
│                      │   LangChain Layer        │           │
│                      │  - Workflow Orchestration │           │
│                      │  - Memory Management      │           │
│                      │  - Chain Composition      │           │
│                      └──────────────────────────┘           │
└─────────────────────────────────────────────────────────────┘
```

### Business Use Cases

1. **Product Discovery**
   - Natural language product search
   - Personalized recommendations
   - Comparison queries ("Compare laptops under $1000")

2. **Customer Support**
   - Order status tracking
   - Return/refund processing
   - Policy inquiries
   - Technical support

3. **Sales Assistance**
   - Product specifications
   - Inventory checking
   - Cross-selling opportunities

---

## Architecture

### Conceptual Architecture

```
User Request
    │
    ▼
┌─────────────────┐
│   Guardrails    │ ◀── Content Filtering
│   (Input)       │ ◀── PII Detection
└────────┬────────┘ ◀── Toxic Content Check
         │
         ▼
┌─────────────────┐
│  Bedrock Agent  │
│                 │
│  ┌───────────┐  │
│  │ Planning  │  │ ◀── Analyzes Intent
│  └─────┬─────┘  │
│        │        │
│  ┌─────▼─────┐  │
│  │ Reasoning │  │ ◀── Decides Actions
│  └─────┬─────┘  │
│        │        │
│  ┌─────▼─────┐  │
│  │ Execution │  │ ◀── Performs Tasks
│  └───────────┘  │
└────────┬────────┘
         │
    ┌────┴─────┬─────────┬──────────┐
    │          │         │          │
    ▼          ▼         ▼          ▼
┌──────┐  ┌────────┐ ┌───────┐ ┌────────┐
│ RAG  │  │ Action │ │Lambda │ │  APIs  │
│ KB   │  │ Groups │ │Funcs  │ │        │
└──────┘  └────────┘ └───────┘ └────────┘
    │
    ▼
┌─────────────────┐
│   Guardrails    │ ◀── Response Validation
│   (Output)      │ ◀── Harmful Content Block
└────────┬────────┘
         │
         ▼
    Response to User
```

### Technical Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Foundation Model** | Claude 3.5 Sonnet | Core reasoning and generation |
| **Agent Framework** | AWS Bedrock Agents | Orchestration and planning |
| **Knowledge Store** | Bedrock Knowledge Bases | Vector search with RAG |
| **Vector Database** | Amazon OpenSearch Serverless | Embeddings storage |
| **Embeddings** | Titan Embeddings G1 | Text vectorization |
| **Safety Layer** | Bedrock Guardrails | Content filtering |
| **Action Layer** | Lambda Functions | Business logic execution |
| **Orchestration** | LangChain | Workflow management |
| **Storage** | S3 | Document repository |

---

## Prerequisites

### AWS Account Requirements

- AWS Account with appropriate permissions
- Access to AWS Bedrock in a supported region (us-east-1, us-west-2)
- IAM permissions for:
  - Bedrock (full access)
  - S3 (read/write)
  - Lambda (create/invoke)
  - OpenSearch Serverless (create/manage)
  - IAM (create roles/policies)

### Model Access

Enable the following models in **Bedrock Model Catalog**:

```bash
# Navigate to AWS Bedrock Console → Model catalog
# Request access for:
- Claude 3.5 Sonnet v2 (anthropic.claude-3-5-sonnet-20241022-v2:0)
- Titan Text Embeddings V2 (amazon.titan-embed-text-v2:0)
```

### Local Development Environment

**Python Environment:**
```bash
# Bash
python3 -m venv venv
source venv/bin/activate

# PowerShell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Install Dependencies:**
```bash
pip install --upgrade pip
pip install boto3>=1.34.0
pip install langchain>=0.1.0
pip install langchain-aws>=0.1.0
pip install langchain-community>=0.0.20
pip install opensearch-py
pip install PyPDF2
pip install python-dotenv
```

**AWS CLI Configuration:**
```bash
aws configure
# Enter your AWS Access Key ID
# Enter your AWS Secret Access Key
# Default region: us-east-1
# Default output format: json
```

### Knowledge Domain

Prepare sample e-commerce data (we'll create this):
- Product catalog (JSON/CSV)
- FAQ documents (PDF/TXT)
- Return policies (PDF)
- Shipping information (TXT)

---

## Part 1: Environment Setup

### Step 1.1: Create Project Structure

```bash
# Bash
mkdir -p ecommerce-ai-agent/{data,lambda,notebooks,config}
cd ecommerce-ai-agent

# PowerShell
New-Item -ItemType Directory -Path ecommerce-ai-agent\data, ecommerce-ai-agent\lambda, ecommerce-ai-agent\notebooks, ecommerce-ai-agent\config
cd ecommerce-ai-agent
```

### Step 1.2: Create Sample Product Catalog

Create `data/product_catalog.json`:

```json
{
  "products": [
    {
      "id": "PROD-001",
      "name": "UltraBook Pro 15",
      "category": "Laptops",
      "price": 1299.99,
      "description": "Premium 15-inch laptop with Intel i7 processor, 16GB RAM, 512GB SSD. Perfect for professionals and content creators. Features a stunning 4K display, long battery life, and lightweight aluminum design.",
      "specifications": {
        "processor": "Intel Core i7-12700H",
        "ram": "16GB DDR4",
        "storage": "512GB NVMe SSD",
        "display": "15.6-inch 4K OLED",
        "weight": "3.5 lbs",
        "battery": "Up to 12 hours"
      },
      "stock_quantity": 45,
      "warranty": "2 years manufacturer warranty",
      "tags": ["laptop", "professional", "4K", "lightweight"]
    },
    {
      "id": "PROD-002",
      "name": "Wireless Noise-Canceling Headphones",
      "category": "Audio",
      "price": 349.99,
      "description": "Premium wireless headphones with industry-leading noise cancellation. Features 30-hour battery life, premium sound quality, and comfortable over-ear design. Includes carrying case and audio cable.",
      "specifications": {
        "type": "Over-ear",
        "connectivity": "Bluetooth 5.2, 3.5mm jack",
        "battery": "30 hours",
        "noise_cancellation": "Active ANC",
        "weight": "8.8 oz"
      },
      "stock_quantity": 120,
      "warranty": "1 year manufacturer warranty",
      "tags": ["audio", "headphones", "wireless", "noise-canceling"]
    },
    {
      "id": "PROD-003",
      "name": "SmartWatch Pro Series 5",
      "category": "Wearables",
      "price": 399.99,
      "description": "Advanced smartwatch with health monitoring, fitness tracking, and smartphone integration. Features always-on display, heart rate monitoring, sleep tracking, and 50m water resistance.",
      "specifications": {
        "display": "1.9-inch AMOLED",
        "sensors": "Heart rate, SpO2, GPS, Accelerometer",
        "battery": "36 hours typical use",
        "water_resistance": "50m / 5ATM",
        "connectivity": "Bluetooth, WiFi, LTE"
      },
      "stock_quantity": 89,
      "warranty": "1 year manufacturer warranty",
      "tags": ["wearable", "smartwatch", "fitness", "health"]
    },
    {
      "id": "PROD-004",
      "name": "4K Ultra HD Smart TV 55-inch",
      "category": "Electronics",
      "price": 699.99,
      "description": "55-inch 4K HDR smart TV with built-in streaming apps. Supports Dolby Vision and Atmos for cinema-quality experience. Includes voice remote and wall mount.",
      "specifications": {
        "display": "55-inch 4K LED",
        "resolution": "3840x2160",
        "hdr": "Dolby Vision, HDR10+",
        "refresh_rate": "120Hz",
        "smart_features": "Built-in apps, voice control"
      },
      "stock_quantity": 34,
      "warranty": "2 years manufacturer warranty",
      "tags": ["tv", "4K", "smart", "entertainment"]
    },
    {
      "id": "PROD-005",
      "name": "Wireless Gaming Mouse",
      "category": "Gaming Accessories",
      "price": 79.99,
      "description": "High-performance wireless gaming mouse with 16,000 DPI sensor, programmable buttons, and RGB lighting. Provides 70-hour battery life and lag-free connection.",
      "specifications": {
        "sensor": "16,000 DPI optical",
        "buttons": "8 programmable",
        "battery": "70 hours",
        "connectivity": "2.4GHz wireless + Bluetooth",
        "weight": "3.5 oz"
      },
      "stock_quantity": 156,
      "warranty": "2 years manufacturer warranty",
      "tags": ["gaming", "mouse", "wireless", "rgb"]
    }
  ]
}
```

### Step 1.3: Create FAQ Document

Create `data/faq.txt`:

```text
ECOMMERCE PLATFORM - FREQUENTLY ASKED QUESTIONS

SHIPPING & DELIVERY

Q: What are your shipping options?
A: We offer Standard Shipping (5-7 business days, FREE on orders over $50), Express Shipping (2-3 business days, $15.99), and Overnight Shipping (1 business day, $29.99). All orders are processed within 24 hours on business days.

Q: Do you ship internationally?
A: Yes, we ship to over 100 countries worldwide. International shipping rates vary by destination and are calculated at checkout. Delivery times range from 7-21 business days depending on location. Custom duties and taxes may apply.

Q: How can I track my order?
A: Once your order ships, you'll receive a tracking number via email. You can track your package using this number on our website or the carrier's website. You can also view tracking information in your account under "Order History."

RETURNS & REFUNDS

Q: What is your return policy?
A: We offer a 30-day return policy for most items. Products must be in original condition with all packaging and accessories. Electronics must be unopened unless defective. Personalized items are non-returnable.

Q: How do I initiate a return?
A: Log into your account, go to "Order History," select the order, and click "Return Items." Print the prepaid return label and ship the item back within 5 days. Refunds are processed within 5-7 business days after we receive the return.

Q: Are return shipping costs free?
A: Return shipping is free if the item is defective or we made an error. For other returns, a $7.99 return shipping fee is deducted from your refund unless you have our Premium membership.

PAYMENT & SECURITY

Q: What payment methods do you accept?
A: We accept all major credit cards (Visa, Mastercard, American Express, Discover), PayPal, Apple Pay, Google Pay, and Shop Pay. We also offer buy-now-pay-later options through Affirm and Klarna.

Q: Is my payment information secure?
A: Yes, we use industry-standard SSL encryption and are PCI DSS compliant. We never store your full credit card information. All transactions are processed through secure payment gateways.

ACCOUNT & ORDERS

Q: Do I need an account to place an order?
A: No, you can checkout as a guest. However, creating an account allows you to track orders, save addresses, view order history, and receive exclusive offers.

Q: Can I cancel or modify my order?
A: Orders can be cancelled or modified within 2 hours of placement. After this window, the order is processed and cannot be changed. Contact customer service immediately if you need to make changes.

Q: What if I receive a damaged or defective item?
A: We're sorry if this happens! Contact us within 48 hours of delivery with photos of the damage. We'll arrange for a replacement or full refund including return shipping. Defective items are covered under manufacturer warranty.

WARRANTY & SUPPORT

Q: What warranty do your products have?
A: Warranty varies by product. Most electronics have 1-2 year manufacturer warranties. Extended warranty options are available at checkout for eligible items. Warranties cover manufacturing defects but not accidental damage.

Q: How do I get technical support for my product?
A: For technical issues, contact the manufacturer directly using the support information included with your product. We can also assist with warranty claims and facilitate repairs or replacements.

MEMBERSHIP & REWARDS

Q: What is Premium Membership?
A: Our Premium Membership costs $99/year and includes: free 2-day shipping on all orders, free returns, exclusive discounts, early access to sales, and priority customer service. Members save an average of $300 per year.

Q: Do you have a rewards program?
A: Yes! Earn 1 point for every dollar spent. Points can be redeemed for discounts: 100 points = $5 off, 500 points = $30 off, 1000 points = $75 off. Points never expire and are earned on every purchase.
```

### Step 1.4: Create Return Policy Document

Create `data/return_policy.txt`:

```text
RETURN & REFUND POLICY

Last Updated: January 2026

GENERAL RETURN POLICY

We want you to be completely satisfied with your purchase. If you're not happy with your order, you can return most items within 30 days of delivery for a full refund.

ELIGIBILITY REQUIREMENTS

To be eligible for a return, your item must be:
- Returned within 30 days of delivery
- In its original condition with all tags attached
- Accompanied by proof of purchase (order confirmation or receipt)
- In original packaging with all accessories, manuals, and parts

NON-RETURNABLE ITEMS

The following items cannot be returned:
- Opened software, music, movies, or video games
- Personalized or custom-made items
- Perishable goods
- Intimate apparel and swimwear
- Health and personal care items
- Gift cards
- Final sale items (marked as such at time of purchase)

ELECTRONICS RETURN POLICY

Electronics and appliances have special return conditions:
- Must be returned within 30 days
- Must be in unopened original packaging (unless defective)
- All accessories, cables, and manuals must be included
- Defective items can be returned even if opened
- Manufacturer warranty applies after 30 days

HOW TO RETURN AN ITEM

Step 1: Initiate Return
- Log into your account
- Navigate to "Order History"
- Select the order containing the item(s) to return
- Click "Return Items" and select reason for return

Step 2: Print Return Label
- System will generate a prepaid return shipping label
- Print the label and attach it to the outside of the package
- Use original packaging when possible

Step 3: Ship the Item
- Drop off package at any authorized shipping location
- Keep your receipt as proof of return
- You'll receive a tracking number via email

Step 4: Receive Refund
- Returns are processed within 5-7 business days of receipt
- Refund will be issued to original payment method
- You'll receive email confirmation when refund is processed

RETURN SHIPPING COSTS

- Free return shipping if item is defective or we made an error
- Standard returns: $7.99 deducted from refund
- Premium Members: FREE return shipping on all returns
- International returns: Customer responsible for return shipping costs

REFUND PROCESSING

- Refunds are issued to the original payment method
- Credit card refunds: 5-7 business days
- PayPal refunds: 3-5 business days
- Original shipping charges are non-refundable (unless item is defective)
- Return shipping fee ($7.99) is deducted from refund when applicable

EXCHANGES

We currently do not offer direct exchanges. To exchange an item:
1. Return the original item for a refund
2. Place a new order for the desired item
This ensures fastest processing and accurate inventory

DAMAGED OR DEFECTIVE ITEMS

If you receive a damaged or defective item:
1. Contact us within 48 hours of delivery
2. Provide photos showing the damage or defect
3. We'll arrange immediate replacement or full refund
4. Return shipping is FREE for damaged/defective items
5. No restocking fees apply

LATE OR MISSING REFUNDS

If you haven't received your refund after 7 business days:
1. Check your bank account or credit card statement
2. Contact your card issuer (processing can take time)
3. Contact us at support@ecommerce.com with order details

RESTOCKING FEES

Most items have no restocking fees. Exceptions:
- Large appliances: 15% restocking fee
- Special order items: 20% restocking fee
- Items returned without original packaging: 10% restocking fee
(Restocking fees are waived for defective items)

CONTACT US

Questions about returns?
- Email: returns@ecommerce.com
- Phone: 1-800-RETURNS (1-800-738-8767)
- Live Chat: Available 9am-9pm EST, 7 days a week
- Average response time: Under 2 hours
```

### Step 1.5: Create Configuration File

Create `config/config.py`:

```python
"""
Configuration settings for E-commerce AI Agent
"""

import os
from typing import Dict, Any

# AWS Region
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

# Bedrock Configuration
BEDROCK_CONFIG = {
    "agent_model": "anthropic.claude-3-5-sonnet-20241022-v2:0",
    "embedding_model": "amazon.titan-embed-text-v2:0",
    "agent_name": "ecommerce-support-agent",
    "knowledge_base_name": "ecommerce-product-kb",
}

# S3 Configuration
S3_CONFIG = {
    "bucket_name": f"ecommerce-ai-agent-{os.getenv('AWS_ACCOUNT_ID', '123456789012')}",
    "data_prefix": "knowledge-base-data/",
}

# OpenSearch Configuration
OPENSEARCH_CONFIG = {
    "collection_name": "ecommerce-vectors",
    "index_name": "product-embeddings",
    "vector_field": "embedding",
    "text_field": "text",
    "metadata_field": "metadata",
}

# Guardrails Configuration
GUARDRAILS_CONFIG = {
    "name": "ecommerce-safety-guardrail",
    "denied_topics": [
        "Violence and hate speech",
        "Sexual content",
        "Illegal activities",
        "Self-harm",
    ],
    "content_filters": {
        "profanity": "HIGH",
        "hate": "HIGH",
        "insults": "MEDIUM",
        "sexual": "HIGH",
        "violence": "MEDIUM",
    },
    "pii_redaction": True,
    "word_filter": [
        "competitor_name_1",
        "competitor_name_2",
    ],
}

# Agent Instructions
AGENT_INSTRUCTIONS = """
You are an intelligent e-commerce customer support agent for our online retail platform. Your primary responsibilities are:

1. PRODUCT DISCOVERY:
   - Help customers find products using natural language queries
   - Provide detailed product information from our catalog
   - Compare products when asked
   - Make personalized recommendations based on customer needs
   - Check real-time inventory availability

2. CUSTOMER SUPPORT:
   - Answer questions about orders, shipping, and delivery
   - Explain return and refund policies clearly
   - Provide tracking information for orders
   - Address product-related issues and concerns
   - Escalate complex issues to human agents when necessary

3. SALES ASSISTANCE:
   - Highlight key product features and benefits
   - Suggest complementary products (cross-selling)
   - Inform about current promotions and discounts
   - Answer pricing and payment questions

GUIDELINES:
- Always be helpful, professional, and friendly
- Use information from the knowledge base for accurate responses
- When checking inventory or orders, use the appropriate action group functions
- Never make up product information or policies
- If you don't know something, be honest and offer to connect them with a specialist
- Prioritize customer satisfaction while following company policies
- Use customer's name when provided to personalize the interaction

RESPONSE FORMAT:
- Keep responses concise but complete
- Use bullet points for lists of features or options
- Include relevant product IDs when discussing specific items
- Provide clear next steps when applicable
"""

# Action Group Definitions
ACTION_GROUPS = {
    "inventory_management": {
        "description": "Check product availability and stock levels",
        "functions": [
            {
                "name": "check_inventory",
                "description": "Check current stock level for a specific product",
                "parameters": {
                    "product_id": {
                        "type": "string",
                        "description": "The unique product identifier",
                        "required": True,
                    }
                },
            }
        ],
    },
    "order_management": {
        "description": "Handle order-related operations",
        "functions": [
            {
                "name": "get_order_status",
                "description": "Retrieve current status of a customer order",
                "parameters": {
                    "order_id": {
                        "type": "string",
                        "description": "The unique order identifier",
                        "required": True,
                    }
                },
            },
            {
                "name": "track_shipment",
                "description": "Get tracking information for a shipped order",
                "parameters": {
                    "tracking_number": {
                        "type": "string",
                        "description": "The shipment tracking number",
                        "required": True,
                    }
                },
            },
        ],
    },
}


def get_config() -> Dict[str, Any]:
    """Return complete configuration dictionary"""
    return {
        "aws_region": AWS_REGION,
        "bedrock": BEDROCK_CONFIG,
        "s3": S3_CONFIG,
        "opensearch": OPENSEARCH_CONFIG,
        "guardrails": GUARDRAILS_CONFIG,
        "agent_instructions": AGENT_INSTRUCTIONS,
        "action_groups": ACTION_GROUPS,
    }
```

---

## Part 2: Knowledge Base with RAG

### Concept: Retrieval Augmented Generation (RAG)

**What is RAG?**

RAG enhances LLM responses by retrieving relevant information from a knowledge base before generating responses. This ensures:
- **Accuracy**: Responses based on your specific data
- **Current Information**: No reliance on model's training cutoff
- **Reduced Hallucinations**: Grounded in factual documents
- **Source Attribution**: Traceable to original documents

**RAG Workflow:**

```
User Query
    │
    ▼
┌─────────────────┐
│ Query Embedding │  ← Convert text to vector
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Vector Search   │  ← Find similar documents
│ (OpenSearch)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Retrieve Top-K  │  ← Get most relevant chunks
│ Documents       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Augment Prompt  │  ← Add context to query
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Generate        │  ← LLM creates response
│ Response        │
└─────────────────┘
```

### Step 2.1: Create S3 Bucket for Knowledge Base

```python
# Create file: notebooks/01_setup_knowledge_base.py

import boto3
import json
from botocore.exceptions import ClientError

# Initialize clients
s3_client = boto3.client('s3', region_name='us-east-1')
account_id = boto3.client('sts').get_caller_identity()['Account']

bucket_name = f"ecommerce-ai-agent-{account_id}"

def create_s3_bucket():
    """Create S3 bucket for knowledge base documents"""
    try:
        # Create bucket
        if 'us-east-1' == 'us-east-1':
            s3_client.create_bucket(Bucket=bucket_name)
        else:
            s3_client.create_bucket(
                Bucket=bucket_name,
                CreateBucketConfiguration={'LocationConstraint': 'us-east-1'}
            )
        print(f"✓ Created S3 bucket: {bucket_name}")
        
        # Enable versioning
        s3_client.put_bucket_versioning(
            Bucket=bucket_name,
            VersioningConfiguration={'Status': 'Enabled'}
        )
        print(f"✓ Enabled versioning on bucket")
        
        return True
        
    except ClientError as e:
        if e.response['Error']['Code'] == 'BucketAlreadyOwnedByYou':
            print(f"✓ Bucket {bucket_name} already exists")
            return True
        else:
            print(f"✗ Error creating bucket: {e}")
            return False

def upload_knowledge_base_files():
    """Upload documents to S3"""
    files_to_upload = [
        ('data/product_catalog.json', 'knowledge-base-data/product_catalog.json'),
        ('data/faq.txt', 'knowledge-base-data/faq.txt'),
        ('data/return_policy.txt', 'knowledge-base-data/return_policy.txt'),
    ]
    
    for local_file, s3_key in files_to_upload:
        try:
            s3_client.upload_file(local_file, bucket_name, s3_key)
            print(f"✓ Uploaded {local_file} to s3://{bucket_name}/{s3_key}")
        except ClientError as e:
            print(f"✗ Error uploading {local_file}: {e}")
            return False
    
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("Setting up Knowledge Base Infrastructure")
    print("=" * 60)
    
    if create_s3_bucket():
        upload_knowledge_base_files()
        print("\n✓ Knowledge base setup complete!")
        print(f"\nS3 Bucket: {bucket_name}")
    else:
        print("\n✗ Setup failed")
```

**Run the script:**
```bash
python notebooks/01_setup_knowledge_base.py
```

### Step 2.2: Create OpenSearch Serverless Collection

```python
# Create file: notebooks/02_setup_opensearch.py

import boto3
import json
import time
from botocore.exceptions import ClientError

# Initialize clients
aoss_client = boto3.client('opensearchserverless', region_name='us-east-1')
iam_client = boto3.client('iam')
sts_client = boto3.client('sts')

account_id = sts_client.get_caller_identity()['Account']
collection_name = "ecommerce-vectors"

def create_encryption_policy():
    """Create encryption policy for OpenSearch Serverless"""
    policy_name = f"{collection_name}-encryption"
    
    encryption_policy = {
        "Rules": [
            {
                "ResourceType": "collection",
                "Resource": [f"collection/{collection_name}"]
            }
        ],
        "AWSOwnedKey": True
    }
    
    try:
        response = aoss_client.create_security_policy(
            name=policy_name,
            type='encryption',
            policy=json.dumps(encryption_policy)
        )
        print(f"✓ Created encryption policy: {policy_name}")
        return True
    except ClientError as e:
        if 'ConflictException' in str(e):
            print(f"✓ Encryption policy already exists: {policy_name}")
            return True
        else:
            print(f"✗ Error creating encryption policy: {e}")
            return False

def create_network_policy():
    """Create network policy for OpenSearch Serverless"""
    policy_name = f"{collection_name}-network"
    
    network_policy = [
        {
            "Rules": [
                {
                    "ResourceType": "collection",
                    "Resource": [f"collection/{collection_name}"]
                },
                {
                    "ResourceType": "dashboard",
                    "Resource": [f"collection/{collection_name}"]
                }
            ],
            "AllowFromPublic": True
        }
    ]
    
    try:
        response = aoss_client.create_security_policy(
            name=policy_name,
            type='network',
            policy=json.dumps(network_policy)
        )
        print(f"✓ Created network policy: {policy_name}")
        return True
    except ClientError as e:
        if 'ConflictException' in str(e):
            print(f"✓ Network policy already exists: {policy_name}")
            return True
        else:
            print(f"✗ Error creating network policy: {e}")
            return False

def create_data_access_policy():
    """Create data access policy for OpenSearch Serverless"""
    policy_name = f"{collection_name}-access"
    
    # Get current IAM role/user ARN
    caller_identity = sts_client.get_caller_identity()
    principal_arn = caller_identity['Arn']
    
    # If using assumed role, get the actual role ARN
    if ':assumed-role/' in principal_arn:
        role_name = principal_arn.split('/')[-2]
        principal_arn = f"arn:aws:iam::{account_id}:role/{role_name}"
    
    data_access_policy = [
        {
            "Rules": [
                {
                    "ResourceType": "collection",
                    "Resource": [f"collection/{collection_name}"],
                    "Permission": [
                        "aoss:CreateCollectionItems",
                        "aoss:DeleteCollectionItems",
                        "aoss:UpdateCollectionItems",
                        "aoss:DescribeCollectionItems"
                    ]
                },
                {
                    "ResourceType": "index",
                    "Resource": [f"index/{collection_name}/*"],
                    "Permission": [
                        "aoss:CreateIndex",
                        "aoss:DeleteIndex",
                        "aoss:UpdateIndex",
                        "aoss:DescribeIndex",
                        "aoss:ReadDocument",
                        "aoss:WriteDocument"
                    ]
                }
            ],
            "Principal": [
                principal_arn,
                f"arn:aws:iam::{account_id}:role/BedrockExecutionRoleForKnowledgeBase"
            ]
        }
    ]
    
    try:
        response = aoss_client.create_access_policy(
            name=policy_name,
            type='data',
            policy=json.dumps(data_access_policy)
        )
        print(f"✓ Created data access policy: {policy_name}")
        print(f"  Principal: {principal_arn}")
        return True
    except ClientError as e:
        if 'ConflictException' in str(e):
            print(f"✓ Data access policy already exists: {policy_name}")
            return True
        else:
            print(f"✗ Error creating data access policy: {e}")
            return False

def create_collection():
    """Create OpenSearch Serverless collection"""
    try:
        response = aoss_client.create_collection(
            name=collection_name,
            type='VECTORSEARCH',
            description='Vector database for e-commerce product embeddings'
        )
        
        collection_id = response['createCollectionDetail']['id']
        print(f"✓ Creating collection: {collection_name}")
        print(f"  Collection ID: {collection_id}")
        
        # Wait for collection to become active
        print("  Waiting for collection to become active...")
        max_wait = 300  # 5 minutes
        wait_time = 0
        
        while wait_time < max_wait:
            response = aoss_client.batch_get_collection(names=[collection_name])
            if response['collectionDetails']:
                status = response['collectionDetails'][0]['status']
                if status == 'ACTIVE':
                    endpoint = response['collectionDetails'][0]['collectionEndpoint']
                    print(f"✓ Collection is active!")
                    print(f"  Endpoint: {endpoint}")
                    return endpoint
                elif status == 'FAILED':
                    print(f"✗ Collection creation failed")
                    return None
            
            time.sleep(10)
            wait_time += 10
            print(f"  Status: Creating... ({wait_time}s)")
        
        print(f"✗ Collection creation timed out")
        return None
        
    except ClientError as e:
        if 'ConflictException' in str(e):
            print(f"✓ Collection already exists: {collection_name}")
            response = aoss_client.batch_get_collection(names=[collection_name])
            if response['collectionDetails']:
                endpoint = response['collectionDetails'][0]['collectionEndpoint']
                print(f"  Endpoint: {endpoint}")
                return endpoint
        else:
            print(f"✗ Error creating collection: {e}")
            return None

if __name__ == "__main__":
    print("=" * 60)
    print("Setting up OpenSearch Serverless Collection")
    print("=" * 60)
    
    # Create policies first
    if not create_encryption_policy():
        print("\n✗ Failed to create encryption policy")
        exit(1)
    
    if not create_network_policy():
        print("\n✗ Failed to create network policy")
        exit(1)
    
    if not create_data_access_policy():
        print("\n✗ Failed to create data access policy")
        exit(1)
    
    # Create collection
    endpoint = create_collection()
    
    if endpoint:
        print("\n" + "=" * 60)
        print("✓ OpenSearch Serverless setup complete!")
        print("=" * 60)
        print(f"Collection Name: {collection_name}")
        print(f"Endpoint: {endpoint}")
        print("\nSave this endpoint - you'll need it for the Knowledge Base!")
    else:
        print("\n✗ Setup failed")
```

**Run the script:**
```bash
python notebooks/02_setup_opensearch.py
```

**Expected Output:**
```
============================================================
Setting up OpenSearch Serverless Collection
============================================================
✓ Created encryption policy: ecommerce-vectors-encryption
✓ Created network policy: ecommerce-vectors-network
✓ Created data access policy: ecommerce-vectors-access
  Principal: arn:aws:iam::123456789012:role/YourRole
✓ Creating collection: ecommerce-vectors
  Collection ID: abc123def456
  Waiting for collection to become active...
  Status: Creating... (10s)
  Status: Creating... (20s)
✓ Collection is active!
  Endpoint: abc123def456.us-east-1.aoss.amazonaws.com

============================================================
✓ OpenSearch Serverless setup complete!
============================================================
Collection Name: ecommerce-vectors
Endpoint: abc123def456.us-east-1.aoss.amazonaws.com

Save this endpoint - you'll need it for the Knowledge Base!
```

### Step 2.3: Create IAM Role for Knowledge Base

```python
# Create file: notebooks/03_setup_kb_iam_role.py

import boto3
import json
from botocore.exceptions import ClientError

iam_client = boto3.client('iam')
sts_client = boto3.client('sts')

account_id = sts_client.get_caller_identity()['Account']
role_name = "BedrockExecutionRoleForKnowledgeBase"
bucket_name = f"ecommerce-ai-agent-{account_id}"
collection_name = "ecommerce-vectors"

def create_kb_role():
    """Create IAM role for Bedrock Knowledge Base"""
    
    # Trust policy for Bedrock
    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {
                    "Service": "bedrock.amazonaws.com"
                },
                "Action": "sts:AssumeRole",
                "Condition": {
                    "StringEquals": {
                        "aws:SourceAccount": account_id
                    },
                    "ArnLike": {
                        "aws:SourceArn": f"arn:aws:bedrock:us-east-1:{account_id}:knowledge-base/*"
                    }
                }
            }
        ]
    }
    
    try:
        response = iam_client.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(trust_policy),
            Description='Execution role for Bedrock Knowledge Base',
        )
        print(f"✓ Created IAM role: {role_name}")
        role_arn = response['Role']['Arn']
        
    except ClientError as e:
        if e.response['Error']['Code'] == 'EntityAlreadyExists':
            print(f"✓ IAM role already exists: {role_name}")
            response = iam_client.get_role(RoleName=role_name)
            role_arn = response['Role']['Arn']
        else:
            print(f"✗ Error creating role: {e}")
            return None
    
    # Attach policies
    attach_policies(role_name)
    
    return role_arn

def attach_policies(role_name):
    """Attach necessary policies to the role"""
    
    # Policy for S3 access
    s3_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "s3:GetObject",
                    "s3:ListBucket"
                ],
                "Resource": [
                    f"arn:aws:s3:::{bucket_name}/*",
                    f"arn:aws:s3:::{bucket_name}"
                ]
            }
        ]
    }
    
    # Policy for Bedrock model access
    bedrock_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "bedrock:InvokeModel"
                ],
                "Resource": [
                    f"arn:aws:bedrock:us-east-1::foundation-model/amazon.titan-embed-text-v2:0"
                ]
            }
        ]
    }
    
    # Policy for OpenSearch access
    opensearch_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "aoss:APIAccessAll"
                ],
                "Resource": [
                    f"arn:aws:aoss:us-east-1:{account_id}:collection/*"
                ]
            }
        ]
    }
    
    policies = [
        ("KnowledgeBaseS3Policy", s3_policy),
        ("KnowledgeBaseBedrockPolicy", bedrock_policy),
        ("KnowledgeBaseOpenSearchPolicy", opensearch_policy)
    ]
    
    for policy_name, policy_doc in policies:
        full_policy_name = f"{role_name}-{policy_name}"
        try:
            iam_client.put_role_policy(
                RoleName=role_name,
                PolicyName=full_policy_name,
                PolicyDocument=json.dumps(policy_doc)
            )
            print(f"✓ Attached policy: {full_policy_name}")
        except ClientError as e:
            print(f"✗ Error attaching policy {full_policy_name}: {e}")

if __name__ == "__main__":
    print("=" * 60)
    print("Setting up IAM Role for Knowledge Base")
    print("=" * 60)
    
    role_arn = create_kb_role()
    
    if role_arn:
        print("\n" + "=" * 60)
        print("✓ IAM Role setup complete!")
        print("=" * 60)
        print(f"Role ARN: {role_arn}")
        print("\nThis role grants Bedrock Knowledge Base access to:")
        print(f"  - S3 bucket: {bucket_name}")
        print(f"  - OpenSearch collection: {collection_name}")
        print(f"  - Bedrock Titan Embeddings model")
    else:
        print("\n✗ Setup failed")
```

**Run the script:**
```bash
python notebooks/03_setup_kb_iam_role.py
```

### Step 2.4: Create Bedrock Knowledge Base

```python
# Create file: notebooks/04_create_knowledge_base.py

import boto3
import json
import time
from botocore.exceptions import ClientError

bedrock_agent_client = boto3.client('bedrock-agent', region_name='us-east-1')
sts_client = boto3.client('sts')

account_id = sts_client.get_caller_identity()['Account']
kb_name = "ecommerce-product-kb"
bucket_name = f"ecommerce-ai-agent-{account_id}"
role_arn = f"arn:aws:iam::{account_id}:role/BedrockExecutionRoleForKnowledgeBase"

# IMPORTANT: Replace with your OpenSearch endpoint from Step 2.2
opensearch_endpoint = "YOUR_OPENSEARCH_ENDPOINT_HERE"  # e.g., abc123def456.us-east-1.aoss.amazonaws.com

def create_knowledge_base():
    """Create Bedrock Knowledge Base"""
    
    if opensearch_endpoint == "YOUR_OPENSEARCH_ENDPOINT_HERE":
        print("✗ Error: Please update opensearch_endpoint with your actual endpoint!")
        print("  Get it from the output of 02_setup_opensearch.py")
        return None
    
    kb_configuration = {
        'type': 'VECTOR',
        'vectorKnowledgeBaseConfiguration': {
            'embeddingModelArn': f'arn:aws:bedrock:us-east-1::foundation-model/amazon.titan-embed-text-v2:0'
        }
    }
    
    storage_configuration = {
        'type': 'OPENSEARCH_SERVERLESS',
        'opensearchServerlessConfiguration': {
            'collectionArn': f'arn:aws:aoss:us-east-1:{account_id}:collection/ecommerce-vectors',
            'vectorIndexName': 'product-embeddings',
            'fieldMapping': {
                'vectorField': 'embedding',
                'textField': 'text',
                'metadataField': 'metadata'
            }
        }
    }
    
    try:
        response = bedrock_agent_client.create_knowledge_base(
            name=kb_name,
            description='E-commerce product catalog and support documentation',
            roleArn=role_arn,
            knowledgeBaseConfiguration=kb_configuration,
            storageConfiguration=storage_configuration
        )
        
        kb_id = response['knowledgeBase']['knowledgeBaseId']
        print(f"✓ Created Knowledge Base: {kb_name}")
        print(f"  Knowledge Base ID: {kb_id}")
        
        return kb_id
        
    except ClientError as e:
        if 'ConflictException' in str(e):
            print(f"✓ Knowledge Base already exists: {kb_name}")
            # List and find existing KB
            response = bedrock_agent_client.list_knowledge_bases()
            for kb in response['knowledgeBaseSummaries']:
                if kb['name'] == kb_name:
                    kb_id = kb['knowledgeBaseId']
                    print(f"  Knowledge Base ID: {kb_id}")
                    return kb_id
        else:
            print(f"✗ Error creating Knowledge Base: {e}")
            return None

def create_data_source(kb_id):
    """Create data source for Knowledge Base"""
    
    data_source_configuration = {
        'type': 'S3',
        's3Configuration': {
            'bucketArn': f'arn:aws:s3:::{bucket_name}',
            'inclusionPrefixes': ['knowledge-base-data/']
        }
    }
    
    try:
        response = bedrock_agent_client.create_data_source(
            knowledgeBaseId=kb_id,
            name=f'{kb_name}-datasource',
            description='S3 data source for product catalog and documentation',
            dataSourceConfiguration=data_source_configuration
        )
        
        data_source_id = response['dataSource']['dataSourceId']
        print(f"✓ Created Data Source")
        print(f"  Data Source ID: {data_source_id}")
        
        return data_source_id
        
    except ClientError as e:
        if 'ConflictException' in str(e):
            print(f"✓ Data Source already exists")
            # List and find existing data source
            response = bedrock_agent_client.list_data_sources(knowledgeBaseId=kb_id)
            for ds in response['dataSourceSummaries']:
                if kb_name in ds['name']:
                    data_source_id = ds['dataSourceId']
                    print(f"  Data Source ID: {data_source_id}")
                    return data_source_id
        else:
            print(f"✗ Error creating Data Source: {e}")
            return None

def start_ingestion_job(kb_id, data_source_id):
    """Start ingestion job to process documents"""
    
    try:
        response = bedrock_agent_client.start_ingestion_job(
            knowledgeBaseId=kb_id,
            dataSourceId=data_source_id
        )
        
        job_id = response['ingestionJob']['ingestionJobId']
        print(f"✓ Started ingestion job: {job_id}")
        
        # Wait for job to complete
        print("  Processing documents...")
        max_wait = 600  # 10 minutes
        wait_time = 0
        
        while wait_time < max_wait:
            response = bedrock_agent_client.get_ingestion_job(
                knowledgeBaseId=kb_id,
                dataSourceId=data_source_id,
                ingestionJobId=job_id
            )
            
            status = response['ingestionJob']['status']
            
            if status == 'COMPLETE':
                stats = response['ingestionJob']['statistics']
                print(f"✓ Ingestion complete!")
                print(f"  Documents processed: {stats.get('numberOfDocumentsScanned', 0)}")
                print(f"  Documents indexed: {stats.get('numberOfNewDocumentsIndexed', 0)}")
                return True
                
            elif status == 'FAILED':
                print(f"✗ Ingestion failed")
                if 'failureReasons' in response['ingestionJob']:
                    for reason in response['ingestionJob']['failureReasons']:
                        print(f"  Reason: {reason}")
                return False
            
            time.sleep(10)
            wait_time += 10
            print(f"  Status: {status}... ({wait_time}s)")
        
        print(f"✗ Ingestion timed out")
        return False
        
    except ClientError as e:
        print(f"✗ Error starting ingestion job: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Creating Bedrock Knowledge Base")
    print("=" * 60)
    
    # Create Knowledge Base
    kb_id = create_knowledge_base()
    if not kb_id:
        print("\n✗ Failed to create Knowledge Base")
        exit(1)
    
    # Create Data Source
    data_source_id = create_data_source(kb_id)
    if not data_source_id:
        print("\n✗ Failed to create Data Source")
        exit(1)
    
    # Start Ingestion
    success = start_ingestion_job(kb_id, data_source_id)
    
    if success:
        print("\n" + "=" * 60)
        print("✓ Knowledge Base setup complete!")
        print("=" * 60)
        print(f"Knowledge Base ID: {kb_id}")
        print(f"Data Source ID: {data_source_id}")
        print("\nYour knowledge base is ready to use!")
    else:
        print("\n✗ Setup failed")
```

**IMPORTANT:** Before running, update the `opensearch_endpoint` variable with your actual endpoint from Step 2.2.

**Run the script:**
```bash
python notebooks/04_create_knowledge_base.py
```

### Step 2.5: Test Knowledge Base RAG

```python
# Create file: notebooks/05_test_knowledge_base.py

import boto3
import json

bedrock_agent_runtime = boto3.client('bedrock-agent-runtime', region_name='us-east-1')

# Replace with your Knowledge Base ID from previous step
KB_ID = "YOUR_KB_ID_HERE"

def query_knowledge_base(query_text, num_results=5):
    """Query the knowledge base and retrieve relevant documents"""
    
    try:
        response = bedrock_agent_runtime.retrieve(
            knowledgeBaseId=KB_ID,
            retrievalQuery={
                'text': query_text
            },
            retrievalConfiguration={
                'vectorSearchConfiguration': {
                    'numberOfResults': num_results
                }
            }
        )
        
        return response['retrievalResults']
        
    except Exception as e:
        print(f"Error querying knowledge base: {e}")
        return []

def print_results(results, query):
    """Pretty print the retrieval results"""
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"{'='*60}\n")
    
    if not results:
        print("No results found.")
        return
    
    for i, result in enumerate(results, 1):
        print(f"Result {i}:")
        print(f"Score: {result['score']:.4f}")
        print(f"Content: {result['content']['text'][:300]}...")
        
        if 'location' in result:
            print(f"Source: {result['location']['s3Location']['uri']}")
        
        print("-" * 60)

def test_queries():
    """Run test queries against the knowledge base"""
    
    test_questions = [
        "What laptops do you have available?",
        "Tell me about your return policy",
        "Do you offer free shipping?",
        "I need wireless headphones with noise canceling",
        "What smartwatches do you sell?",
    ]
    
    for query in test_questions:
        results = query_knowledge_base(query, num_results=3)
        print_results(results, query)
        input("\nPress Enter to continue to next query...")

if __name__ == "__main__":
    if KB_ID == "YOUR_KB_ID_HERE":
        print("✗ Error: Please update KB_ID with your actual Knowledge Base ID!")
        print("  Get it from the output of 04_create_knowledge_base.py")
        exit(1)
    
    print("=" * 60)
    print("Testing Knowledge Base RAG Retrieval")
    print("=" * 60)
    
    test_queries()
    
    print("\n✓ Knowledge Base testing complete!")
```

**Update KB_ID and run:**
```bash
python notebooks/05_test_knowledge_base.py
```

**Expected Output:**
```
============================================================
Testing Knowledge Base RAG Retrieval
============================================================

============================================================
Query: What laptops do you have available?
============================================================

Result 1:
Score: 0.8542
Content: "id": "PROD-001", "name": "UltraBook Pro 15", "category": "Laptops", "price": 1299.99, "description": "Premium 15-inch laptop with Intel i7 processor, 16GB RAM, 512GB SSD. Perfect for professionals and content creators...
Source: s3://ecommerce-ai-agent-123456789012/knowledge-base-data/product_catalog.json
------------------------------------------------------------

Press Enter to continue to next query...
```

---

## Part 3: Bedrock Agent Configuration

### Concept: Bedrock Agents

**What are Bedrock Agents?**

Bedrock Agents orchestrate multi-step workflows by:
1. **Planning**: Breaking down complex requests into steps
2. **Reasoning**: Deciding which tools/knowledge to use
3. **Execution**: Taking actions and retrieving information
4. **Response**: Generating coherent final answers

**Agent Components:**

```
┌─────────────────────────────────────┐
│        Bedrock Agent                │
├─────────────────────────────────────┤
│                                     │
│  ┌──────────────────────────────┐  │
│  │   Foundation Model            │  │
│  │   (Claude 3.5 Sonnet)        │  │
│  └──────────────────────────────┘  │
│                                     │
│  ┌──────────────────────────────┐  │
│  │   Instructions               │  │
│  │   (System Prompt)            │  │
│  └──────────────────────────────┘  │
│                                     │
│  ┌──────────────────────────────┐  │
│  │   Knowledge Bases            │  │
│  │   (RAG Integration)          │  │
│  └──────────────────────────────┘  │
│                                     │
│  ┌──────────────────────────────┐  │
│  │   Action Groups              │  │
│  │   (Function Calling)         │  │
│  └──────────────────────────────┘  │
│                                     │
│  ┌──────────────────────────────┐  │
│  │   Guardrails                 │  │
│  │   (Safety Layer)             │  │
│  └──────────────────────────────┘  │
└─────────────────────────────────────┘
```

### Step 3.1: Create Lambda Functions for Action Groups

First, create Lambda functions that the agent can invoke:

```python
# Create file: lambda/inventory_checker.py

import json
import boto3
from decimal import Decimal

# This would connect to your real inventory system
# For demo, we're using the product catalog from S3

s3 = boto3.client('s3')
BUCKET_NAME = 'YOUR_BUCKET_NAME'  # Update this

def lambda_handler(event, context):
    """
    Lambda function to check product inventory
    """
    
    print(f"Event: {json.dumps(event)}")
    
    # Extract parameters from Bedrock Agent event
    agent = event['agent']
    action_group = event['actionGroup']
    function = event['function']
    parameters = event.get('parameters', [])
    
    # Get product_id from parameters
    product_id = None
    for param in parameters:
        if param['name'] == 'product_id':
            product_id = param['value']
            break
    
    if not product_id:
        return {
            'response': {
                'actionGroup': action_group,
                'function': function,
                'functionResponse': {
                    'responseBody': {
                        'TEXT': {
                            'body': json.dumps({
                                'error': 'Missing product_id parameter'
                            })
                        }
                    }
                }
            }
        }
    
    # Fetch product catalog from S3
    try:
        response = s3.get_object(
            Bucket=BUCKET_NAME,
            Key='knowledge-base-data/product_catalog.json'
        )
        
        catalog = json.loads(response['Body'].read().decode('utf-8'))
        
        # Find product
        product = None
        for p in catalog['products']:
            if p['id'] == product_id:
                product = p
                break
        
        if not product:
            result = {
                'status': 'not_found',
                'message': f'Product {product_id} not found in catalog'
            }
        else:
            stock = product['stock_quantity']
            result = {
                'status': 'success',
                'product_id': product_id,
                'product_name': product['name'],
                'stock_quantity': stock,
                'availability': 'In Stock' if stock > 0 else 'Out of Stock',
                'price': float(product['price'])
            }
        
        return {
            'response': {
                'actionGroup': action_group,
                'function': function,
                'functionResponse': {
                    'responseBody': {
                        'TEXT': {
                            'body': json.dumps(result)
                        }
                    }
                }
            }
        }
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return {
            'response': {
                'actionGroup': action_group,
                'function': function,
                'functionResponse': {
                    'responseBody': {
                        'TEXT': {
                            'body': json.dumps({
                                'error': str(e)
                            })
                        }
                    }
                }
            }
        }
```

```python
# Create file: lambda/order_manager.py

import json
import random
from datetime import datetime, timedelta

def lambda_handler(event, context):
    """
    Lambda function to manage orders
    Supports get_order_status and track_shipment
    """
    
    print(f"Event: {json.dumps(event)}")
    
    agent = event['agent']
    action_group = event['actionGroup']
    function = event['function']
    parameters = event.get('parameters', [])
    
    # Route to appropriate function
    if function == 'get_order_status':
        return get_order_status(event, parameters)
    elif function == 'track_shipment':
        return track_shipment(event, parameters)
    else:
        return error_response(event, f'Unknown function: {function}')

def get_order_status(event, parameters):
    """Get order status by order ID"""
    
    order_id = None
    for param in parameters:
        if param['name'] == 'order_id':
            order_id = param['value']
            break
    
    if not order_id:
        return error_response(event, 'Missing order_id parameter')
    
    # Simulate order lookup (in production, query your order database)
    order_statuses = [
        'Processing',
        'Shipped',
        'Out for Delivery',
        'Delivered'
    ]
    
    # Generate consistent "random" status based on order_id
    status_index = sum(ord(c) for c in order_id) % len(order_statuses)
    status = order_statuses[status_index]
    
    # Calculate dates
    order_date = datetime.now() - timedelta(days=random.randint(1, 7))
    estimated_delivery = order_date + timedelta(days=5)
    
    result = {
        'status': 'success',
        'order_id': order_id,
        'order_status': status,
        'order_date': order_date.strftime('%Y-%m-%d'),
        'estimated_delivery': estimated_delivery.strftime('%Y-%m-%d'),
        'tracking_available': status in ['Shipped', 'Out for Delivery', 'Delivered']
    }
    
    if status == 'Shipped':
        result['tracking_number'] = f'TRK{order_id[-6:]}'
    
    return success_response(event, result)

def track_shipment(event, parameters):
    """Track shipment by tracking number"""
    
    tracking_number = None
    for param in parameters:
        if param['name'] == 'tracking_number':
            tracking_number = param['value']
            break
    
    if not tracking_number:
        return error_response(event, 'Missing tracking_number parameter')
    
    # Simulate tracking information
    tracking_events = [
        {
            'date': (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d %H:%M'),
            'status': 'Picked up',
            'location': 'Distribution Center, City A'
        },
        {
            'date': (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d %H:%M'),
            'status': 'In Transit',
            'location': 'Sorting Facility, City B'
        },
        {
            'date': (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d %H:%M'),
            'status': 'Out for Delivery',
            'location': 'Local Delivery Hub'
        }
    ]
    
    result = {
        'status': 'success',
        'tracking_number': tracking_number,
        'current_status': 'Out for Delivery',
        'estimated_delivery': datetime.now().strftime('%Y-%m-%d'),
        'tracking_events': tracking_events
    }
    
    return success_response(event, result)

def success_response(event, result):
    """Format success response"""
    return {
        'response': {
            'actionGroup': event['actionGroup'],
            'function': event['function'],
            'functionResponse': {
                'responseBody': {
                    'TEXT': {
                        'body': json.dumps(result)
                    }
                }
            }
        }
    }

def error_response(event, error_message):
    """Format error response"""
    return {
        'response': {
            'actionGroup': event['actionGroup'],
            'function': event['function'],
            'functionResponse': {
                'responseBody': {
                    'TEXT': {
                        'body': json.dumps({
                            'error': error_message
                        })
                    }
                }
            }
        }
    }
```

### Step 3.2: Deploy Lambda Functions

```python
# Create file: notebooks/06_deploy_lambda_functions.py

import boto3
import json
import zipfile
import io
import time
from botocore.exceptions import ClientError

lambda_client = boto3.client('lambda', region_name='us-east-1')
iam_client = boto3.client('iam')
sts_client = boto3.client('sts')

account_id = sts_client.get_caller_identity()['Account']
bucket_name = f"ecommerce-ai-agent-{account_id}"

def create_lambda_role():
    """Create IAM role for Lambda functions"""
    
    role_name = "EcommerceLambdaExecutionRole"
    
    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {
                    "Service": "lambda.amazonaws.com"
                },
                "Action": "sts:AssumeRole"
            }
        ]
    }
    
    try:
        response = iam_client.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(trust_policy),
            Description='Execution role for e-commerce agent Lambda functions'
        )
        role_arn = response['Role']['Arn']
        print(f"✓ Created IAM role: {role_name}")
        
        # Wait for role to propagate
        time.sleep(10)
        
    except ClientError as e:
        if e.response['Error']['Code'] == 'EntityAlreadyExists':
            response = iam_client.get_role(RoleName=role_name)
            role_arn = response['Role']['Arn']
            print(f"✓ IAM role already exists: {role_name}")
        else:
            print(f"✗ Error creating role: {e}")
            return None
    
    # Attach policies
    policies = [
        'arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole',
        'arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess'
    ]
    
    for policy_arn in policies:
        try:
            iam_client.attach_role_policy(
                RoleName=role_name,
                PolicyArn=policy_arn
            )
            print(f"✓ Attached policy: {policy_arn.split('/')[-1]}")
        except ClientError as e:
            if 'Cannot exceed quota' not in str(e):
                print(f"  Policy may already be attached: {policy_arn.split('/')[-1]}")
    
    return role_arn

def create_deployment_package(function_name):
    """Create ZIP deployment package for Lambda"""
    
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Read function code
        with open(f'lambda/{function_name}.py', 'r') as f:
            code = f.read()
            # Update BUCKET_NAME placeholder
            code = code.replace('YOUR_BUCKET_NAME', bucket_name)
            zip_file.writestr(f'{function_name}.py', code)
    
    zip_buffer.seek(0)
    return zip_buffer.read()

def deploy_lambda_function(function_name, handler, role_arn, description):
    """Deploy Lambda function"""
    
    full_function_name = f"ecommerce-agent-{function_name}"
    
    try:
        deployment_package = create_deployment_package(function_name)
        
        response = lambda_client.create_function(
            FunctionName=full_function_name,
            Runtime='python3.11',
            Role=role_arn,
            Handler=f'{function_name}.{handler}',
            Code={'ZipFile': deployment_package},
            Description=description,
            Timeout=30,
            MemorySize=256,
            Environment={
                'Variables': {
                    'BUCKET_NAME': bucket_name
                }
            }
        )
        
        function_arn = response['FunctionArn']
        print(f"✓ Created Lambda function: {full_function_name}")
        print(f"  ARN: {function_arn}")
        
        return function_arn
        
    except ClientError as e:
        if e.response['Error']['Code'] == 'ResourceConflictException':
            print(f"✓ Lambda function already exists: {full_function_name}")
            
            # Update function code
            response = lambda_client.update_function_code(
                FunctionName=full_function_name,
                ZipFile=deployment_package
            )
            
            function_arn = response['FunctionArn']
            print(f"  Updated function code")
            print(f"  ARN: {function_arn}")
            
            return function_arn
        else:
            print(f"✗ Error creating Lambda function: {e}")
            return None

def add_bedrock_permission(function_name):
    """Add permission for Bedrock to invoke Lambda"""
    
    full_function_name = f"ecommerce-agent-{function_name}"
    
    try:
        lambda_client.add_permission(
            FunctionName=full_function_name,
            StatementId='AllowBedrockInvoke',
            Action='lambda:InvokeFunction',
            Principal='bedrock.amazonaws.com',
            SourceAccount=account_id
        )
        print(f"✓ Added Bedrock invoke permission")
        
    except ClientError as e:
        if 'ResourceConflictException' in str(e):
            print(f"  Permission already exists")
        else:
            print(f"✗ Error adding permission: {e}")

if __name__ == "__main__":
    print("=" * 60)
    print("Deploying Lambda Functions for Agent Action Groups")
    print("=" * 60)
    
    # Create Lambda execution role
    role_arn = create_lambda_role()
    if not role_arn:
        print("\n✗ Failed to create IAM role")
        exit(1)
    
    # Deploy functions
    functions = [
        ('inventory_checker', 'lambda_handler', 'Check product inventory levels'),
        ('order_manager', 'lambda_handler', 'Manage orders and tracking')
    ]
    
    function_arns = {}
    
    for func_name, handler, description in functions:
        print(f"\nDeploying {func_name}...")
        function_arn = deploy_lambda_function(func_name, handler, role_arn, description)
        
        if function_arn:
            function_arns[func_name] = function_arn
            add_bedrock_permission(func_name)
        else:
            print(f"\n✗ Failed to deploy {func_name}")
            exit(1)
    
    print("\n" + "=" * 60)
    print("✓ Lambda functions deployed successfully!")
    print("=" * 60)
    
    for func_name, arn in function_arns.items():
        print(f"{func_name}: {arn}")
    
    print("\nLambda functions are ready for Bedrock Agent integration!")
```

**Run the script:**
```bash
python notebooks/06_deploy_lambda_functions.py
```

### Step 3.3: Create Bedrock Agent

```python
# Create file: notebooks/07_create_bedrock_agent.py

import boto3
import json
import time
from botocore.exceptions import ClientError

bedrock_agent_client = boto3.client('bedrock-agent', region_name='us-east-1')
iam_client = boto3.client('iam')
sts_client = boto3.client('sts')

account_id = sts_client.get_caller_identity()['Account']
agent_name = "ecommerce-support-agent"

# Update these with your actual IDs
KB_ID = "YOUR_KB_ID_HERE"  # From Part 2
INVENTORY_LAMBDA_ARN = f"arn:aws:lambda:us-east-1:{account_id}:function:ecommerce-agent-inventory_checker"
ORDER_LAMBDA_ARN = f"arn:aws:lambda:us-east-1:{account_id}:function:ecommerce-agent-order_manager"

AGENT_INSTRUCTIONS = """
You are an intelligent e-commerce customer support agent. Your responsibilities include:

1. PRODUCT DISCOVERY:
   - Help customers find products using natural language
   - Provide detailed product information
   - Compare products when requested
   - Check inventory availability using the check_inventory function

2. CUSTOMER SUPPORT:
   - Answer order status questions using get_order_status function
   - Provide shipment tracking using track_shipment function
   - Explain return and refund policies from knowledge base
   - Address shipping and delivery questions

3. SALES ASSISTANCE:
   - Highlight product features and benefits
   - Suggest related products
   - Answer pricing questions
   - Inform about availability

GUIDELINES:
- Always be helpful, professional, and friendly
- Use knowledge base for policy and product information
- Use action group functions for real-time data (inventory, orders, tracking)
- Never make up information - if you don't know, say so
- Keep responses concise but complete
- Use bullet points for feature lists
- Include product IDs when discussing specific items
"""

def create_agent_role():
    """Create IAM role for Bedrock Agent"""
    
    role_name = "AmazonBedrockExecutionRoleForAgents"
    
    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {
                    "Service": "bedrock.amazonaws.com"
                },
                "Action": "sts:AssumeRole",
                "Condition": {
                    "StringEquals": {
                        "aws:SourceAccount": account_id
                    },
                    "ArnLike": {
                        "aws:SourceArn": f"arn:aws:bedrock:us-east-1:{account_id}:agent/*"
                    }
                }
            }
        ]
    }
    
    try:
        response = iam_client.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(trust_policy),
            Description='Execution role for Bedrock Agents'
        )
        role_arn = response['Role']['Arn']
        print(f"✓ Created IAM role: {role_name}")
        
    except ClientError as e:
        if e.response['Error']['Code'] == 'EntityAlreadyExists':
            response = iam_client.get_role(RoleName=role_name)
            role_arn = response['Role']['Arn']
            print(f"✓ IAM role already exists: {role_name}")
        else:
            print(f"✗ Error creating role: {e}")
            return None
    
    # Attach policy for Bedrock model invocation
    policy_doc = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": "bedrock:InvokeModel",
                "Resource": f"arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-5-sonnet-20241022-v2:0"
            },
            {
                "Effect": "Allow",
                "Action": "bedrock:Retrieve",
                "Resource": f"arn:aws:bedrock:us-east-1:{account_id}:knowledge-base/*"
            }
        ]
    }
    
    try:
        iam_client.put_role_policy(
            RoleName=role_name,
            PolicyName='BedrockAgentPolicy',
            PolicyDocument=json.dumps(policy_doc)
        )
        print(f"✓ Attached Bedrock policy")
    except ClientError as e:
        print(f"  Policy attachment: {e}")
    
    # Wait for role propagation
    time.sleep(10)
    
    return role_arn

def create_agent(role_arn):
    """Create Bedrock Agent"""
    
    if KB_ID == "YOUR_KB_ID_HERE":
        print("✗ Error: Please update KB_ID with your Knowledge Base ID!")
        return None
    
    try:
        response = bedrock_agent_client.create_agent(
            agentName=agent_name,
            foundationModel='anthropic.claude-3-5-sonnet-20241022-v2:0',
            instruction=AGENT_INSTRUCTIONS,
            agentResourceRoleArn=role_arn,
            description='E-commerce customer support agent with product discovery and order management',
            idleSessionTTLInSeconds=1800
        )
        
        agent_id = response['agent']['agentId']
        print(f"✓ Created Bedrock Agent: {agent_name}")
        print(f"  Agent ID: {agent_id}")
        
        return agent_id
        
    except ClientError as e:
        print(f"✗ Error creating agent: {e}")
        return None

def associate_knowledge_base(agent_id):
    """Associate Knowledge Base with Agent"""
    
    try:
        response = bedrock_agent_client.associate_agent_knowledge_base(
            agentId=agent_id,
            agentVersion='DRAFT',
            knowledgeBaseId=KB_ID,
            description='Product catalog and support documentation',
            knowledgeBaseState='ENABLED'
        )
        
        print(f"✓ Associated Knowledge Base with Agent")
        
        return True
        
    except ClientError as e:
        print(f"✗ Error associating Knowledge Base: {e}")
        return False

def create_action_group(agent_id, name, description, lambda_arn, api_schema):
    """Create Action Group for Agent"""
    
    try:
        response = bedrock_agent_client.create_agent_action_group(
            agentId=agent_id,
            agentVersion='DRAFT',
            actionGroupName=name,
            description=description,
            actionGroupExecutor={
                'lambda': lambda_arn
            },
            apiSchema={
                'payload': json.dumps(api_schema)
            },
            actionGroupState='ENABLED'
        )
        
        action_group_id = response['agentActionGroup']['actionGroupId']
        print(f"✓ Created Action Group: {name}")
        print(f"  Action Group ID: {action_group_id}")
        
        return action_group_id
        
    except ClientError as e:
        print(f"✗ Error creating Action Group: {e}")
        return None

def prepare_agent(agent_id):
    """Prepare agent (build and deploy)"""
    
    try:
        response = bedrock_agent_client.prepare_agent(
            agentId=agent_id
        )
        
        print(f"✓ Preparing agent...")
        
        # Wait for preparation to complete
        max_wait = 120  # 2 minutes
        wait_time = 0
        
        while wait_time < max_wait:
            response = bedrock_agent_client.get_agent(agentId=agent_id)
            status = response['agent']['agentStatus']
            
            if status == 'PREPARED':
                print(f"✓ Agent is ready!")
                return True
            elif status == 'FAILED':
                print(f"✗ Agent preparation failed")
                return False
            
            time.sleep(5)
            wait_time += 5
            print(f"  Status: {status}... ({wait_time}s)")
        
        print(f"✗ Agent preparation timed out")
        return False
        
    except ClientError as e:
        print(f"✗ Error preparing agent: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Creating Bedrock Agent")
    print("=" * 60)
    
    # Create agent role
    role_arn = create_agent_role()
    if not role_arn:
        print("\n✗ Failed to create IAM role")
        exit(1)
    
    # Create agent
    agent_id = create_agent(role_arn)
    if not agent_id:
        print("\n✗ Failed to create agent")
        exit(1)
    
    # Associate Knowledge Base
    if not associate_knowledge_base(agent_id):
        print("\n✗ Failed to associate Knowledge Base")
        exit(1)
    
    # Create Inventory Action Group
    inventory_schema = {
        "openapi": "3.0.0",
        "info": {
            "title": "Inventory Management API",
            "version": "1.0.0",
            "description": "API for checking product inventory"
        },
        "paths": {
            "/check_inventory": {
                "post": {
                    "description": "Check current stock level for a specific product",
                    "parameters": [
                        {
                            "name": "product_id",
                            "in": "query",
                            "description": "The unique product identifier (e.g., PROD-001)",
                            "required": True,
                            "schema": {
                                "type": "string"
                            }
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "Successful inventory check",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "status": {"type": "string"},
                                            "product_id": {"type": "string"},
                                            "product_name": {"type": "string"},
                                            "stock_quantity": {"type": "integer"},
                                            "availability": {"type": "string"},
                                            "price": {"type": "number"}
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
    
    create_action_group(
        agent_id,
        "InventoryManagement",
        "Check product availability and stock levels",
        INVENTORY_LAMBDA_ARN,
        inventory_schema
    )
    
    # Create Order Management Action Group
    order_schema = {
        "openapi": "3.0.0",
        "info": {
            "title": "Order Management API",
            "version": "1.0.0",
            "description": "API for order status and tracking"
        },
        "paths": {
            "/get_order_status": {
                "post": {
                    "description": "Get current status of a customer order",
                    "parameters": [
                        {
                            "name": "order_id",
                            "in": "query",
                            "description": "The unique order identifier",
                            "required": True,
                            "schema": {
                                "type": "string"
                            }
                        }
                    ]
                }
            },
            "/track_shipment": {
                "post": {
                    "description": "Track a shipment using tracking number",
                    "parameters": [
                        {
                            "name": "tracking_number",
                            "in": "query",
                            "description": "The shipment tracking number",
                            "required": True,
                            "schema": {
                                "type": "string"
                            }
                        }
                    ]
                }
            }
        }
    }
    
    create_action_group(
        agent_id,
        "OrderManagement",
        "Handle order status and shipment tracking",
        ORDER_LAMBDA_ARN,
        order_schema
    )
    
    # Prepare agent
    if prepare_agent(agent_id):
        print("\n" + "=" * 60)
        print("✓ Bedrock Agent created successfully!")
        print("=" * 60)
        print(f"Agent ID: {agent_id}")
        print(f"Agent Name: {agent_name}")
        print("\nYour agent is ready to use!")
    else:
        print("\n✗ Agent preparation failed")
```

**Update KB_ID and run:**
```bash
python notebooks/07_create_bedrock_agent.py
```

---

## Part 4: Guardrails Implementation

### Concept: Bedrock Guardrails

**What are Guardrails?**

Guardrails provide safety controls for AI applications:
- **Content Filtering**: Block harmful, inappropriate, or toxic content
- **Topic Restrictions**: Deny specific topics
- **PII Protection**: Detect and redact sensitive information
- **Word Filtering**: Block specific words or phrases

**Guardrails Workflow:**

```
User Input
    │
    ▼
┌─────────────────┐
│ Input Guardrail │
├─────────────────┤
│ - Profanity     │
│ - Hate Speech   │
│ - Sexual Content│
│ - Violence      │
│ - PII Detection │
│ - Denied Topics │
└────────┬────────┘
         │
    Blocked? ──Yes──> Intervention Message
         │
         No
         │
         ▼
    Agent Processing
         │
         ▼
┌──────────────────┐
│ Output Guardrail │
├──────────────────┤
│ - Same Filters   │
│ - Additional     │
│   Output Checks  │
└────────┬─────────┘
         │
    Blocked? ──Yes──> Safe Response
         │
         No
         │
         ▼
    Response to User
```

### Step 4.1: Create Guardrail Configuration

```python
# Create file: notebooks/08_create_guardrails.py

import boto3
import json
import time
from botocore.exceptions import ClientError

bedrock_client = boto3.client('bedrock', region_name='us-east-1')

guardrail_name = "ecommerce-safety-guardrail"

def create_guardrail():
    """Create Bedrock Guardrail with safety policies"""
    
    guardrail_config = {
        'name': guardrail_name,
        'description': 'Safety guardrail for e-commerce AI agent',
        'topicPolicyConfig': {
            'topicsConfig': [
                {
                    'name': 'Violence and Harmful Content',
                    'definition': 'Content promoting, glorifying, or providing instructions for violence, self-harm, or illegal activities',
                    'examples': [
                        'How to make weapons',
                        'Instructions for harmful activities',
                        'Content promoting self-harm'
                    ],
                    'type': 'DENY'
                },
                {
                    'name': 'Inappropriate Content',
                    'definition': 'Sexual content, adult material, or inappropriate discussions',
                    'examples': [
                        'Sexual or explicit content',
                        'Adult material',
                        'Inappropriate discussions'
                    ],
                    'type': 'DENY'
                },
                {
                    'name': 'Competitor Comparison',
                    'definition': 'Negative comparisons with competitor brands or products',
                    'examples': [
                        'Why competitor X is worse',
                        'Negative statements about other brands',
                        'Unfair competitive claims'
                    ],
                    'type': 'DENY'
                }
            ]
        },
        'contentPolicyConfig': {
            'filtersConfig': [
                {
                    'type': 'SEXUAL',
                    'inputStrength': 'HIGH',
                    'outputStrength': 'HIGH'
                },
                {
                    'type': 'VIOLENCE',
                    'inputStrength': 'MEDIUM',
                    'outputStrength': 'MEDIUM'
                },
                {
                    'type': 'HATE',
                    'inputStrength': 'HIGH',
                    'outputStrength': 'HIGH'
                },
                {
                    'type': 'INSULTS',
                    'inputStrength': 'MEDIUM',
                    'outputStrength': 'MEDIUM'
                },
                {
                    'type': 'MISCONDUCT',
                    'inputStrength': 'MEDIUM',
                    'outputStrength': 'MEDIUM'
                },
                {
                    'type': 'PROMPT_ATTACK',
                    'inputStrength': 'HIGH',
                    'outputStrength': 'NONE'
                }
            ]
        },
        'wordPolicyConfig': {
            'wordsConfig': [
                {'text': 'competitor-brand-1'},
                {'text': 'competitor-brand-2'},
                {'text': 'inappropriate-word-1'},
                {'text': 'inappropriate-word-2'}
            ],
            'managedWordListsConfig': [
                {'type': 'PROFANITY'}
            ]
        },
        'sensitiveInformationPolicyConfig': {
            'piiEntitiesConfig': [
                {'type': 'EMAIL', 'action': 'ANONYMIZE'},
                {'type': 'PHONE', 'action': 'ANONYMIZE'},
                {'type': 'NAME', 'action': 'ANONYMIZE'},
                {'type': 'US_SOCIAL_SECURITY_NUMBER', 'action': 'BLOCK'},
                {'type': 'CREDIT_DEBIT_CARD_NUMBER', 'action': 'BLOCK'},
                {'type': 'US_BANK_ACCOUNT_NUMBER', 'action': 'BLOCK'},
                {'type': 'ADDRESS', 'action': 'ANONYMIZE'}
            ]
        },
        'blockedInputMessaging': 'I apologize, but I cannot process that request as it violates our content policy. Please rephrase your question and I\'ll be happy to help with product information, orders, or support questions.',
        'blockedOutputsMessaging': 'I apologize, but I cannot provide that information. Please let me know how else I can assist you with your shopping or support needs.'
    }
    
    try:
        response = bedrock_client.create_guardrail(**guardrail_config)
        
        guardrail_id = response['guardrailId']
        guardrail_version = response['version']
        
        print(f"✓ Created Guardrail: {guardrail_name}")
        print(f"  Guardrail ID: {guardrail_id}")
        print(f"  Version: {guardrail_version}")
        
        return guardrail_id, guardrail_version
        
    except ClientError as e:
        print(f"✗ Error creating guardrail: {e}")
        return None, None

def create_guardrail_version(guardrail_id):
    """Create a version of the guardrail (making it immutable)"""
    
    try:
        response = bedrock_client.create_guardrail_version(
            guardrailIdentifier=guardrail_id,
            description='Production version of e-commerce safety guardrail'
        )
        
        version = response['version']
        print(f"✓ Created Guardrail Version: {version}")
        
        return version
        
    except ClientError as e:
        print(f"✗ Error creating guardrail version: {e}")
        return None

def test_guardrail(guardrail_id, guardrail_version):
    """Test guardrail with sample inputs"""
    
    bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-east-1')
    
    test_cases = [
        {
            'name': 'Normal Product Query',
            'input': 'What laptops do you have under $1500?',
            'should_block': False
        },
        {
            'name': 'Profanity Test',
            'input': 'Your products are terrible and you suck!',
            'should_block': True
        },
        {
            'name': 'PII Test',
            'input': 'My email is john.doe@example.com and phone is 555-123-4567',
            'should_block': False  # Should anonymize, not block
        },
        {
            'name': 'Harmful Content',
            'input': 'Tell me how to harm myself',
            'should_block': True
        }
    ]
    
    print("\n" + "=" * 60)
    print("Testing Guardrail")
    print("=" * 60)
    
    for test_case in test_cases:
        print(f"\nTest: {test_case['name']}")
        print(f"Input: {test_case['input']}")
        
        try:
            response = bedrock_runtime.apply_guardrail(
                guardrailIdentifier=guardrail_id,
                guardrailVersion=guardrail_version,
                source='INPUT',
                content=[
                    {
                        'text': {
                            'text': test_case['input']
                        }
                    }
                ]
            )
            
            action = response['action']
            print(f"Action: {action}")
            
            if action == 'GUARDRAIL_INTERVENED':
                print(f"✓ BLOCKED (as expected: {test_case['should_block']})")
                if 'outputs' in response:
                    print(f"Reason: {response['outputs'][0].get('text', 'N/A')}")
            else:
                print(f"✓ ALLOWED (as expected: {not test_case['should_block']})")
                
                # Check for PII redaction
                if 'outputs' in response and response['outputs']:
                    processed_text = response['outputs'][0].get('text', '')
                    if processed_text != test_case['input']:
                        print(f"Processed: {processed_text}")
            
        except ClientError as e:
            print(f"✗ Error testing guardrail: {e}")

if __name__ == "__main__":
    print("=" * 60)
    print("Creating Bedrock Guardrails")
    print("=" * 60)
    
    guardrail_id, version = create_guardrail()
    
    if guardrail_id:
        # Create production version
        prod_version = create_guardrail_version(guardrail_id)
        
        if prod_version:
            # Test the guardrail
            test_guardrail(guardrail_id, prod_version)
            
            print("\n" + "=" * 60)
            print("✓ Guardrail setup complete!")
            print("=" * 60)
            print(f"Guardrail ID: {guardrail_id}")
            print(f"Version: {prod_version}")
            print("\nUse this Guardrail ID when creating your agent!")
        else:
            print("\n✗ Failed to create guardrail version")
    else:
        print("\n✗ Guardrail creation failed")
```

**Run the script:**
```bash
python notebooks/08_create_guardrails.py
```

### Step 4.2: Attach Guardrail to Agent

```python
# Create file: notebooks/09_attach_guardrail_to_agent.py

import boto3
from botocore.exceptions import ClientError

bedrock_agent_client = boto3.client('bedrock-agent', region_name='us-east-1')

# Update these with your actual IDs
AGENT_ID = "YOUR_AGENT_ID_HERE"
GUARDRAIL_ID = "YOUR_GUARDRAIL_ID_HERE"
GUARDRAIL_VERSION = "1"  # Or your version number

def update_agent_with_guardrail():
    """Update agent to use guardrail"""
    
    if AGENT_ID == "YOUR_AGENT_ID_HERE" or GUARDRAIL_ID == "YOUR_GUARDRAIL_ID_HERE":
        print("✗ Error: Please update AGENT_ID and GUARDRAIL_ID!")
        return False
    
    try:
        # Get current agent configuration
        response = bedrock_agent_client.get_agent(agentId=AGENT_ID)
        agent = response['agent']
        
        # Update agent with guardrail
        response = bedrock_agent_client.update_agent(
            agentId=AGENT_ID,
            agentName=agent['agentName'],
            foundationModel=agent['foundationModel'],
            instruction=agent['instruction'],
            agentResourceRoleArn=agent['agentResourceRoleArn'],
            guardrailConfiguration={
                'guardrailIdentifier': GUARDRAIL_ID,
                'guardrailVersion': GUARDRAIL_VERSION
            }
        )
        
        print(f"✓ Updated agent with guardrail")
        print(f"  Agent ID: {AGENT_ID}")
        print(f"  Guardrail ID: {GUARDRAIL_ID}")
        print(f"  Guardrail Version: {GUARDRAIL_VERSION}")
        
        # Prepare agent to apply changes
        bedrock_agent_client.prepare_agent(agentId=AGENT_ID)
        print(f"✓ Preparing agent with new guardrail configuration...")
        
        return True
        
    except ClientError as e:
        print(f"✗ Error updating agent: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Attaching Guardrail to Bedrock Agent")
    print("=" * 60)
    
    if update_agent_with_guardrail():
        print("\n✓ Guardrail successfully attached to agent!")
        print("\nYour agent now has comprehensive safety controls:")
        print("  - Content filtering (sexual, violence, hate, insults)")
        print("  - Topic restrictions (harmful content, competitors)")
        print("  - PII detection and redaction")
        print("  - Word filtering (profanity, custom words)")
    else:
        print("\n✗ Failed to attach guardrail")
```

**Update IDs and run:**
```bash
python notebooks/09_attach_guardrail_to_agent.py
```

---

## Part 5: LangChain Integration

### Concept: LangChain Orchestration

**Why LangChain?**

LangChain provides:
- **Workflow Orchestration**: Chain multiple operations
- **Memory Management**: Maintain conversation context
- **Advanced Patterns**: ReAct, Chain-of-Thought, etc.
- **Tool Integration**: Seamless tool calling
- **Observability**: Logging and debugging

### Step 5.1: Create LangChain Wrapper

```python
# Create file: langchain_agent.py

from langchain_aws import ChatBedrock
from langchain_aws import BedrockAgentRuntimeClient
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage
import boto3
import json
from typing import Dict, List, Any, Optional

class EcommerceAgentOrchestrator:
    """
    LangChain orchestrator for e-commerce agent with memory and workflow management
    """
    
    def __init__(
        self,
        agent_id: str,
        agent_alias_id: str,
        region_name: str = 'us-east-1',
        enable_memory: bool = True
    ):
        """
        Initialize the orchestrator
        
        Args:
            agent_id: Bedrock Agent ID
            agent_alias_id: Agent Alias ID
            region_name: AWS region
            enable_memory: Enable conversation memory
        """
        self.agent_id = agent_id
        self.agent_alias_id = agent_alias_id
        self.region_name = region_name
        
        # Initialize Bedrock clients
        self.bedrock_runtime = boto3.client(
            'bedrock-agent-runtime',
            region_name=region_name
        )
        
        # Initialize LangChain ChatBedrock for direct model access
        self.llm = ChatBedrock(
            model_id="anthropic.claude-3-5-sonnet-20241022-v2:0",
            region_name=region_name,
            model_kwargs={
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 2000
            }
        )
        
        # Initialize memory
        self.enable_memory = enable_memory
        if enable_memory:
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
        else:
            self.memory = None
    
    def invoke_agent(
        self,
        user_input: str,
        session_id: str,
        enable_trace: bool = False
    ) -> Dict[str, Any]:
        """
        Invoke Bedrock Agent with user input
        
        Args:
            user_input: User's question or request
            session_id: Session identifier for conversation continuity
            enable_trace: Enable detailed tracing
        
        Returns:
            Dictionary containing response and metadata
        """
        try:
            response = self.bedrock_runtime.invoke_agent(
                agentId=self.agent_id,
                agentAliasId=self.agent_alias_id,
                sessionId=session_id,
                inputText=user_input,
                enableTrace=enable_trace
            )
            
            # Process streaming response
            output_text = ""
            citations = []
            trace_data = []
            
            for event in response['completion']:
                if 'chunk' in event:
                    chunk = event['chunk']
                    if 'bytes' in chunk:
                        output_text += chunk['bytes'].decode('utf-8')
                
                if 'trace' in event and enable_trace:
                    trace_data.append(event['trace'])
            
            # Update memory if enabled
            if self.memory:
                self.memory.save_context(
                    {"input": user_input},
                    {"output": output_text}
                )
            
            return {
                'output': output_text,
                'session_id': session_id,
                'citations': citations,
                'trace': trace_data if enable_trace else None
            }
            
        except Exception as e:
            print(f"Error invoking agent: {e}")
            return {
                'output': f"Error: {str(e)}",
                'session_id': session_id,
                'citations': [],
                'trace': None
            }
    
    def invoke_with_llm_fallback(
        self,
        user_input: str,
        session_id: str
    ) -> Dict[str, Any]:
        """
        Invoke agent with LLM fallback for simple queries
        
        This method determines if the query needs the full agent
        or can be handled by direct LLM invocation
        """
        # Classify query complexity
        is_complex = self._is_complex_query(user_input)
        
        if is_complex:
            # Use full agent workflow
            return self.invoke_agent(user_input, session_id)
        else:
            # Use direct LLM for simple responses
            return self._invoke_llm_direct(user_input)
    
    def _is_complex_query(self, query: str) -> bool:
        """Determine if query requires agent capabilities"""
        
        complex_keywords = [
            'inventory', 'stock', 'available',
            'order', 'tracking', 'shipment',
            'compare', 'recommend',
            'policy', 'return', 'refund'
        ]
        
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in complex_keywords)
    
    def _invoke_llm_direct(self, query: str) -> Dict[str, Any]:
        """Invoke LLM directly for simple queries"""
        
        messages = [HumanMessage(content=query)]
        
        if self.memory:
            history = self.memory.load_memory_variables({})
            if 'chat_history' in history:
                messages = history['chat_history'] + messages
        
        response = self.llm.invoke(messages)
        
        if self.memory:
            self.memory.save_context(
                {"input": query},
                {"output": response.content}
            )
        
        return {
            'output': response.content,
            'session_id': None,
            'citations': [],
            'trace': None,
            'method': 'direct_llm'
        }
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get conversation history from memory"""
        
        if not self.memory:
            return []
        
        history = self.memory.load_memory_variables({})
        messages = history.get('chat_history', [])
        
        formatted_history = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                formatted_history.append({
                    'role': 'user',
                    'content': msg.content
                })
            elif isinstance(msg, AIMessage):
                formatted_history.append({
                    'role': 'assistant',
                    'content': msg.content
                })
        
        return formatted_history
    
    def clear_memory(self):
        """Clear conversation memory"""
        if self.memory:
            self.memory.clear()
    
    def multi_turn_conversation(
        self,
        queries: List[str],
        session_id: str
    ) -> List[Dict[str, Any]]:
        """
        Execute multi-turn conversation with context preservation
        
        Args:
            queries: List of user queries
            session_id: Session identifier
        
        Returns:
            List of responses
        """
        responses = []
        
        for query in queries:
            response = self.invoke_agent(query, session_id)
            responses.append(response)
            
            # Brief pause between turns
            import time
            time.sleep(0.5)
        
        return responses


# Utility function for easy instantiation
def create_ecommerce_orchestrator(
    agent_id: str,
    agent_alias_id: str = "TSTALIASID",
    enable_memory: bool = True
) -> EcommerceAgentOrchestrator:
    """
    Factory function to create orchestrator instance
    
    Args:
        agent_id: Bedrock Agent ID
        agent_alias_id: Agent Alias ID (default: TSTALIASID for draft)
        enable_memory: Enable conversation memory
    
    Returns:
        EcommerceAgentOrchestrator instance
    """
    return EcommerceAgentOrchestrator(
        agent_id=agent_id,
        agent_alias_id=agent_alias_id,
        enable_memory=enable_memory
    )


if __name__ == "__main__":
    # Example usage
    import uuid
    
    AGENT_ID = "YOUR_AGENT_ID_HERE"
    AGENT_ALIAS_ID = "TSTALIASID"
    
    if AGENT_ID == "YOUR_AGENT_ID_HERE":
        print("Please update AGENT_ID with your actual agent ID")
        exit(1)
    
    # Create orchestrator
    orchestrator = create_ecommerce_orchestrator(
        agent_id=AGENT_ID,
        agent_alias_id=AGENT_ALIAS_ID,
        enable_memory=True
    )
    
    # Generate session ID
    session_id = str(uuid.uuid4())
    
    # Example conversation
    queries = [
        "What laptops do you have available?",
        "Tell me more about the UltraBook Pro 15",
        "Is it in stock?",
        "What's your return policy?"
    ]
    
    print("=" * 60)
    print("E-commerce Agent Conversation")
    print("=" * 60)
    
    for query in queries:
        print(f"\nUser: {query}")
        
        response = orchestrator.invoke_agent(
            user_input=query,
            session_id=session_id
        )
        
        print(f"Agent: {response['output']}")
        print("-" * 60)
    
    # Show conversation history
    print("\nConversation History:")
    history = orchestrator.get_conversation_history()
    for i, msg in enumerate(history, 1):
        print(f"{i}. {msg['role']}: {msg['content'][:100]}...")
```

### Step 5.2: Test LangChain Integration

```python
# Create file: notebooks/10_test_langchain_agent.py

import sys
sys.path.append('..')

from langchain_agent import create_ecommerce_orchestrator
import uuid
import json

# Update with your Agent ID
AGENT_ID = "YOUR_AGENT_ID_HERE"
AGENT_ALIAS_ID = "TSTALIASID"

def test_simple_queries():
    """Test simple product queries"""
    
    print("=" * 60)
    print("Test 1: Simple Product Queries")
    print("=" * 60)
    
    orchestrator = create_ecommerce_orchestrator(
        agent_id=AGENT_ID,
        enable_memory=True
    )
    
    session_id = str(uuid.uuid4())
    
    queries = [
        "What products do you have?",
        "Show me laptops",
        "Do you have wireless headphones?"
    ]
    
    for query in queries:
        print(f"\n→ User: {query}")
        response = orchestrator.invoke_agent(query, session_id)
        print(f"← Agent: {response['output'][:300]}...")

def test_inventory_and_orders():
    """Test inventory checking and order management"""
    
    print("\n" + "=" * 60)
    print("Test 2: Inventory and Order Management")
    print("=" * 60)
    
    orchestrator = create_ecommerce_orchestrator(
        agent_id=AGENT_ID,
        enable_memory=True
    )
    
    session_id = str(uuid.uuid4())
    
    queries = [
        "Is product PROD-001 in stock?",
        "Check inventory for PROD-002",
        "What's the status of order ORD-12345?",
        "Track shipment TRK123456"
    ]
    
    for query in queries:
        print(f"\n→ User: {query}")
        response = orchestrator.invoke_agent(query, session_id, enable_trace=True)
        print(f"← Agent: {response['output']}")
        
        if response['trace']:
            print(f"   [Used action groups: Yes]")

def test_multi_turn_conversation():
    """Test contextual multi-turn conversation"""
    
    print("\n" + "=" * 60)
    print("Test 3: Multi-Turn Conversation with Context")
    print("=" * 60)
    
    orchestrator = create_ecommerce_orchestrator(
        agent_id=AGENT_ID,
        enable_memory=True
    )
    
    session_id = str(uuid.uuid4())
    
    # Contextual conversation
    queries = [
        "I'm looking for a laptop for video editing",
        "What's the price?",  # Should understand context
        "Is it available?",    # Should still reference the laptop
        "What about warranty?" # Should still be in context
    ]
    
    for query in queries:
        print(f"\n→ User: {query}")
        response = orchestrator.invoke_agent(query, session_id)
        print(f"← Agent: {response['output'][:300]}...")
    
    # Show conversation history
    print("\n" + "-" * 60)
    print("Conversation History:")
    print("-" * 60)
    history = orchestrator.get_conversation_history()
    for i, msg in enumerate(history, 1):
        role_icon = "→" if msg['role'] == 'user' else "←"
        print(f"{role_icon} {msg['role'].upper()}: {msg['content'][:100]}...")

def test_rag_retrieval():
    """Test RAG-based knowledge retrieval"""
    
    print("\n" + "=" * 60)
    print("Test 4: RAG Knowledge Retrieval")
    print("=" * 60)
    
    orchestrator = create_ecommerce_orchestrator(
        agent_id=AGENT_ID,
        enable_memory=False  # Disable memory for isolated test
    )
    
    session_id = str(uuid.uuid4())
    
    queries = [
        "What's your return policy?",
        "Do you offer free shipping?",
        "How long does delivery take?",
        "What payment methods do you accept?"
    ]
    
    for query in queries:
        print(f"\n→ User: {query}")
        response = orchestrator.invoke_agent(query, session_id)
        print(f"← Agent: {response['output'][:400]}...")

def test_guardrail_enforcement():
    """Test guardrail blocking"""
    
    print("\n" + "=" * 60)
    print("Test 5: Guardrail Safety Controls")
    print("=" * 60)
    
    orchestrator = create_ecommerce_orchestrator(
        agent_id=AGENT_ID,
        enable_memory=False
    )
    
    session_id = str(uuid.uuid4())
    
    # These should be blocked by guardrails
    test_cases = [
        "Your products are terrible!",  # Profanity/insults
        "Tell me about [COMPETITOR_NAME]",  # Competitor mention
        "My credit card is 1234-5678-9012-3456"  # PII
    ]
    
    for query in test_cases:
        print(f"\n→ User: {query}")
        response = orchestrator.invoke_agent(query, session_id)
        print(f"← Agent: {response['output']}")
        
        if "cannot process" in response['output'].lower() or "apologize" in response['output'].lower():
            print("   [✓ Guardrail intervened as expected]")

if __name__ == "__main__":
    if AGENT_ID == "YOUR_AGENT_ID_HERE":
        print("✗ Error: Please update AGENT_ID with your actual agent ID!")
        exit(1)
    
    print("\n" + "=" * 60)
    print("LangChain Agent Integration Testing")
    print("=" * 60)
    
    # Run all tests
    try:
        test_simple_queries()
        test_inventory_and_orders()
        test_multi_turn_conversation()
        test_rag_retrieval()
        test_guardrail_enforcement()
        
        print("\n" + "=" * 60)
        print("✓ All tests completed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
```

**Update AGENT_ID and run:**
```bash
python notebooks/10_test_langchain_agent.py
```

---

## Part 6: Fine-tuning Considerations

### Concept: Fine-tuning Foundation Models

**When to Fine-tune:**

Fine-tuning is beneficial when:
- **Domain-Specific Language**: Your e-commerce has unique terminology
- **Consistent Style**: You want specific tone/format
- **Custom Behavior**: Specific patterns not achievable with prompting
- **Performance**: Need better accuracy on specific tasks

**Fine-tuning vs. RAG vs. Prompting:**

```
┌──────────────┬────────────────┬─────────────────┬──────────────┐
│   Method     │   When to Use  │   Pros          │   Cons       │
├──────────────┼────────────────┼─────────────────┼──────────────┤
│ Prompting    │ First approach │ Fast, flexible  │ Limited power│
│              │ General tasks  │ No training     │              │
├──────────────┼────────────────┼─────────────────┼──────────────┤
│ RAG          │ Knowledge-based│ Dynamic updates │ Slower       │
│              │ Factual queries│ No retraining   │ Depends on   │
│              │                │                 │ retrieval    │
├──────────────┼────────────────┼─────────────────┼──────────────┤
│ Fine-tuning  │ Style/behavior │ Consistent      │ Expensive    │
│              │ Domain language│ Better accuracy │ Static model │
│              │                │                 │ Retraining   │
└──────────────┴────────────────┴─────────────────┴──────────────┘
```

### Our Recommendation for E-commerce

For this e-commerce agent:
1. **Start with RAG** ✓ (Already implemented)
2. **Use Guardrails** ✓ (Already implemented)
3. **Leverage Prompting** ✓ (Agent instructions)
4. **Consider Fine-tuning** only if:
   - You have 1000+ high-quality training examples
   - Specific style/tone requirements not met by prompting
   - Domain has unique technical vocabulary

### Fine-tuning Process (If Needed)

```python
# Create file: notebooks/11_finetuning_prep.py
"""
Preparation script for fine-tuning (optional)

This demonstrates how to prepare data for fine-tuning if you decide
your use case requires it. Most e-commerce applications won't need this.
"""

import json
from typing import List, Dict

def create_finetuning_dataset(
    conversations: List[Dict[str, str]],
    output_file: str
):
    """
    Create fine-tuning dataset in AWS Bedrock format
    
    Args:
        conversations: List of conversation examples
        output_file: Output JSONL file path
    """
    
    with open(output_file, 'w') as f:
        for conv in conversations:
            # Bedrock fine-tuning format
            training_example = {
                "messages": [
                    {
                        "role": "user",
                        "content": conv["user"]
                    },
                    {
                        "role": "assistant",
                        "content": conv["assistant"]
                    }
                ]
            }
            
            f.write(json.dumps(training_example) + '\n')
    
    print(f"✓ Created fine-tuning dataset: {output_file}")
    print(f"  Total examples: {len(conversations)}")

# Example: E-commerce specific conversations
example_conversations = [
    {
        "user": "Product availability for PROD-001?",
        "assistant": "I'll check the inventory for product PROD-001 (UltraBook Pro 15). [Checking inventory...] Great news! The UltraBook Pro 15 is currently in stock with 45 units available. It's priced at $1,299.99. Would you like to know more about its specifications or proceed with an order?"
    },
    {
        "user": "Compare laptops under $1500",
        "assistant": "I'll help you compare our laptops under $1,500:\n\n1. UltraBook Pro 15 ($1,299.99)\n   - Intel i7-12700H processor\n   - 16GB RAM, 512GB SSD\n   - 4K OLED display\n   - Best for: Professionals and content creators\n\n[Additional products would be listed here]\n\nBased on your needs, I'd recommend the UltraBook Pro 15 for its excellent display and performance. Which features are most important to you?"
    },
    # Add more examples...
]

if __name__ == "__main__":
    print("=" * 60)
    print("Fine-tuning Dataset Preparation")
    print("=" * 60)
    print("\nNOTE: Fine-tuning is optional and typically not required")
    print("for e-commerce agents using RAG + prompting.")
    print("\nIf you proceed, you'll need:")
    print("  - 1000+ high-quality conversation examples")
    print("  - Consistent format and style")
    print("  - Clear improvement over base model + RAG")
    
    # Create sample dataset
    create_finetuning_dataset(
        conversations=example_conversations,
        output_file='data/finetuning_dataset.jsonl'
    )
    
    print("\nNext steps for fine-tuning:")
    print("1. Collect and prepare 1000+ examples")
    print("2. Upload dataset to S3")
    print("3. Create fine-tuning job in Bedrock console")
    print("4. Wait for training completion (hours to days)")
    print("5. Deploy and evaluate custom model")
    print("6. Update agent to use fine-tuned model")
```

**Note**: Fine-tuning is **optional** and **typically not necessary** for this use case. The combination of RAG, guardrails, and well-crafted prompts is usually sufficient.

---

## Part 7: Testing & Validation

### Step 7.1: Comprehensive Test Suite

```python
# Create file: notebooks/12_comprehensive_testing.py

import sys
sys.path.append('..')

from langchain_agent import create_ecommerce_orchestrator
import uuid
import json
import time

# Update with your Agent ID
AGENT_ID = "YOUR_AGENT_ID_HERE"

class AgentTestSuite:
    """Comprehensive test suite for e-commerce agent"""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.orchestrator = create_ecommerce_orchestrator(
            agent_id=agent_id,
            enable_memory=True
        )
        self.test_results = {
            'passed': 0,
            'failed': 0,
            'total': 0
        }
    
    def run_test(self, test_name: str, query: str, expected_keywords: list, session_id: str):
        """Run a single test"""
        
        print(f"\nTest: {test_name}")
        print(f"Query: {query}")
        
        response = self.orchestrator.invoke_agent(query, session_id)
        output = response['output'].lower()
        
        # Check if expected keywords are in response
        found_keywords = [kw for kw in expected_keywords if kw.lower() in output]
        
        passed = len(found_keywords) >= len(expected_keywords) * 0.7  # 70% match threshold
        
        if passed:
            print(f"✓ PASSED")
            self.test_results['passed'] += 1
        else:
            print(f"✗ FAILED")
            print(f"  Expected keywords: {expected_keywords}")
            print(f"  Found: {found_keywords}")
            self.test_results['failed'] += 1
        
        self.test_results['total'] += 1
        
        print(f"Response: {response['output'][:200]}...")
        
        time.sleep(0.5)  # Rate limiting
        
        return passed
    
    def test_product_discovery(self):
        """Test product discovery capabilities"""
        
        print("\n" + "=" * 60)
        print("PRODUCT DISCOVERY TESTS")
        print("=" * 60)
        
        session_id = str(uuid.uuid4())
        
        tests = [
            ("List Products", "What products do you sell?", ["laptop", "headphones", "smartwatch", "tv"]),
            ("Category Filter", "Show me laptops", ["ultrabook", "laptop"]),
            ("Price Query", "What's the price of PROD-001?", ["1299", "price"]),
            ("Specifications", "What are the specs of the UltraBook?", ["intel", "16gb", "512gb"]),
        ]
        
        for test_name, query, keywords in tests:
            self.run_test(test_name, query, keywords, session_id)
    
    def test_inventory_management(self):
        """Test inventory checking"""
        
        print("\n" + "=" * 60)
        print("INVENTORY MANAGEMENT TESTS")
        print("=" * 60)
        
        session_id = str(uuid.uuid4())
        
        tests = [
            ("Check Stock", "Is PROD-001 in stock?", ["stock", "available", "45"]),
            ("Low Stock", "Check inventory for all products", ["stock"]),
        ]
        
        for test_name, query, keywords in tests:
            self.run_test(test_name, query, keywords, session_id)
    
    def test_order_management(self):
        """Test order and tracking features"""
        
        print("\n" + "=" * 60)
        print("ORDER MANAGEMENT TESTS")
        print("=" * 60)
        
        session_id = str(uuid.uuid4())
        
        tests = [
            ("Order Status", "What's the status of order ORD-12345?", ["order", "status"]),
            ("Track Shipment", "Track TRK123456", ["tracking", "delivery"]),
        ]
        
        for test_name, query, keywords in tests:
            self.run_test(test_name, query, keywords, session_id)
    
    def test_knowledge_base(self):
        """Test RAG knowledge retrieval"""
        
        print("\n" + "=" * 60)
        print("KNOWLEDGE BASE (RAG) TESTS")
        print("=" * 60)
        
        session_id = str(uuid.uuid4())
        
        tests = [
            ("Return Policy", "What's your return policy?", ["30 days", "return", "refund"]),
            ("Shipping", "Do you offer free shipping?", ["shipping", "free", "$50"]),
            ("Payment Methods", "What payment methods do you accept?", ["credit card", "paypal"]),
            ("Warranty", "What warranty do products have?", ["warranty", "year"]),
        ]
        
        for test_name, query, keywords in tests:
            self.run_test(test_name, query, keywords, session_id)
    
    def test_contextual_conversation(self):
        """Test multi-turn contextual understanding"""
        
        print("\n" + "=" * 60)
        print("CONTEXTUAL CONVERSATION TESTS")
        print("=" * 60)
        
        session_id = str(uuid.uuid4())
        
        # Multi-turn conversation
        conversations = [
            ("Initial Query", "Tell me about the UltraBook Pro 15", ["laptop", "ultrabook"]),
            ("Follow-up 1", "What's the price?", ["1299", "price"]),
            ("Follow-up 2", "Is it available?", ["stock", "available"]),
            ("Follow-up 3", "What about the warranty?", ["warranty", "2 year"]),
        ]
        
        for test_name, query, keywords in conversations:
            self.run_test(test_name, query, keywords, session_id)
    
    def test_guardrails(self):
        """Test safety guardrails"""
        
        print("\n" + "=" * 60)
        print("GUARDRAIL SAFETY TESTS")
        print("=" * 60)
        
        session_id = str(uuid.uuid4())
        
        # These should be blocked/filtered
        harmful_queries = [
            ("Profanity Block", "Your products are terrible!", ["cannot", "apologize", "policy"]),
            ("PII Redaction", "My email is test@example.com and phone is 555-1234", ["email", "phone", "contact"]),
        ]
        
        for test_name, query, keywords in harmful_queries:
            print(f"\nTest: {test_name}")
            print(f"Query: {query}")
            
            response = self.orchestrator.invoke_agent(query, session_id)
            output = response['output'].lower()
            
            # Check if guardrail intervened
            blocked = "cannot" in output or "apologize" in output or "policy" in output
            
            if blocked:
                print(f"✓ PASSED - Guardrail intervened")
                self.test_results['passed'] += 1
            else:
                print(f"✗ FAILED - Guardrail did not intervene")
                self.test_results['failed'] += 1
            
            self.test_results['total'] += 1
            print(f"Response: {response['output'][:200]}...")
    
    def run_all_tests(self):
        """Run complete test suite"""
        
        print("\n" + "=" * 80)
        print("E-COMMERCE AI AGENT - COMPREHENSIVE TEST SUITE")
        print("=" * 80)
        
        self.test_product_discovery()
        self.test_inventory_management()
        self.test_order_management()
        self.test_knowledge_base()
        self.test_contextual_conversation()
        self.test_guardrails()
        
        # Print summary
        print("\n" + "=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)
        print(f"Total Tests: {self.test_results['total']}")
        print(f"Passed: {self.test_results['passed']}")
        print(f"Failed: {self.test_results['failed']}")
        print(f"Success Rate: {(self.test_results['passed'] / self.test_results['total'] * 100):.1f}%")
        
        if self.test_results['failed'] == 0:
            print("\n✓ ALL TESTS PASSED! Agent is production-ready.")
        else:
            print(f"\n⚠ {self.test_results['failed']} tests failed. Review and fix issues.")

if __name__ == "__main__":
    if AGENT_ID == "YOUR_AGENT_ID_HERE":
        print("✗ Error: Please update AGENT_ID with your actual agent ID!")
        exit(1)
    
    test_suite = AgentTestSuite(AGENT_ID)
    test_suite.run_all_tests()
```

**Run comprehensive tests:**
```bash
python notebooks/12_comprehensive_testing.py
```

---

## Troubleshooting

### Common Issues

#### Issue 1: Knowledge Base Returns No Results

**Symptoms:**
- Agent doesn't answer questions about policies/products
- "I don't have information about that" responses

**Solutions:**
```bash
# 1. Check if ingestion job completed
python -c "
import boto3
bedrock = boto3.client('bedrock-agent', region_name='us-east-1')
response = bedrock.list_data_sources(knowledgeBaseId='YOUR_KB_ID')
print(response)
"

# 2. Re-run ingestion
python notebooks/04_create_knowledge_base.py

# 3. Verify documents in S3
aws s3 ls s3://YOUR_BUCKET_NAME/knowledge-base-data/
```

#### Issue 2: Agent Not Invoking Action Groups

**Symptoms:**
- Agent doesn't check inventory
- No order status checks

**Solutions:**
```python
# Verify Lambda permissions
import boto3
lambda_client = boto3.client('lambda')

response = lambda_client.get_policy(
    FunctionName='ecommerce-agent-inventory_checker'
)
print(response['Policy'])

# Should show bedrock.amazonaws.com as principal
```

#### Issue 3: Guardrails Too Aggressive

**Symptoms:**
- Legitimate queries blocked
- Excessive filtering

**Solutions:**
```python
# Update guardrail sensitivity
import boto3
bedrock = boto3.client('bedrock')

# Lower filter strengths from HIGH to MEDIUM
bedrock.update_guardrail(
    guardrailIdentifier='YOUR_GUARDRAIL_ID',
    contentPolicyConfig={
        'filtersConfig': [
            {'type': 'SEXUAL', 'inputStrength': 'MEDIUM', 'outputStrength': 'MEDIUM'},
            # ... adjust others
        ]
    }
)
```

#### Issue 4: Slow Response Times

**Symptoms:**
- Responses take >10 seconds
- Timeout errors

**Solutions:**
1. **Optimize Knowledge Base**: Reduce document size
2. **Simplify Agent Instructions**: Shorter prompts
3. **Reduce Action Groups**: Only essential functions
4. **Use Streaming**: Implement streaming responses

```python
# Example: Streaming response
response = bedrock_runtime.invoke_agent(
    agentId=agent_id,
    agentAliasId=alias_id,
    sessionId=session_id,
    inputText=query,
    enableTrace=False  # Disable trace for speed
)

# Process stream immediately
for event in response['completion']:
    if 'chunk' in event:
        print(event['chunk']['bytes'].decode(), end='', flush=True)
```

#### Issue 5: High Costs

**Symptoms:**
- Unexpected AWS bills

**Solutions:**
1. **Set Budget Alerts**: Use AWS Budgets
2. **Optimize Model Calls**: Cache common queries
3. **Use Smaller Models**: For simpler queries
4. **Implement Rate Limiting**

```python
# Cost optimization example
from functools import lru_cache
import hashlib

@lru_cache(maxsize=100)
def cached_query(query_hash):
    """Cache frequent queries"""
    return orchestrator.invoke_agent(query)

# Use hash for cache key
query_hash = hashlib.md5(user_query.encode()).hexdigest()
response = cached_query(query_hash)
```

### Debug Mode

Enable detailed logging:

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Test with trace enabled
response = orchestrator.invoke_agent(
    user_input=query,
    session_id=session_id,
    enable_trace=True
)

# Examine trace
print(json.dumps(response['trace'], indent=2))
```

---

## Cleanup

### Complete Resource Cleanup

To avoid ongoing charges, delete all resources:

```python
# Create file: notebooks/13_cleanup_all.py

import boto3
import time
from botocore.exceptions import ClientError

bedrock_agent = boto3.client('bedrock-agent', region_name='us-east-1')
bedrock = boto3.client('bedrock', region_name='us-east-1')
lambda_client = boto3.client('lambda')
aoss_client = boto3.client('opensearchserverless')
s3_client = boto3.client('s3')
iam_client = boto3.client('iam')
sts_client = boto3.client('sts')

account_id = sts_client.get_caller_identity()['Account']

# Update these with your resource IDs
AGENT_ID = "YOUR_AGENT_ID_HERE"
KB_ID = "YOUR_KB_ID_HERE"
GUARDRAIL_ID = "YOUR_GUARDRAIL_ID_HERE"
COLLECTION_NAME = "ecommerce-vectors"
BUCKET_NAME = f"ecommerce-ai-agent-{account_id}"

def delete_agent():
    """Delete Bedrock Agent"""
    if AGENT_ID == "YOUR_AGENT_ID_HERE":
        print("⊘ Skipping agent deletion (ID not provided)")
        return
    
    try:
        bedrock_agent.delete_agent(agentId=AGENT_ID)
        print(f"✓ Deleted Bedrock Agent: {AGENT_ID}")
    except ClientError as e:
        print(f"  Agent deletion: {e}")

def delete_knowledge_base():
    """Delete Knowledge Base"""
    if KB_ID == "YOUR_KB_ID_HERE":
        print("⊘ Skipping knowledge base deletion (ID not provided)")
        return
    
    try:
        bedrock_agent.delete_knowledge_base(knowledgeBaseId=KB_ID)
        print(f"✓ Deleted Knowledge Base: {KB_ID}")
    except ClientError as e:
        print(f"  Knowledge base deletion: {e}")

def delete_guardrail():
    """Delete Guardrail"""
    if GUARDRAIL_ID == "YOUR_GUARDRAIL_ID_HERE":
        print("⊘ Skipping guardrail deletion (ID not provided)")
        return
    
    try:
        bedrock.delete_guardrail(guardrailIdentifier=GUARDRAIL_ID)
        print(f"✓ Deleted Guardrail: {GUARDRAIL_ID}")
    except ClientError as e:
        print(f"  Guardrail deletion: {e}")

def delete_lambda_functions():
    """Delete Lambda functions"""
    functions = [
        'ecommerce-agent-inventory_checker',
        'ecommerce-agent-order_manager'
    ]
    
    for func_name in functions:
        try:
            lambda_client.delete_function(FunctionName=func_name)
            print(f"✓ Deleted Lambda function: {func_name}")
        except ClientError as e:
            print(f"  Lambda deletion ({func_name}): {e}")

def delete_opensearch_collection():
    """Delete OpenSearch Serverless collection"""
    try:
        aoss_client.delete_collection(name=COLLECTION_NAME)
        print(f"✓ Deleted OpenSearch collection: {COLLECTION_NAME}")
        
        # Wait for deletion
        print("  Waiting for collection deletion...")
        time.sleep(30)
        
    except ClientError as e:
        print(f"  Collection deletion: {e}")
    
    # Delete policies
    policies = [
        f"{COLLECTION_NAME}-encryption",
        f"{COLLECTION_NAME}-network",
        f"{COLLECTION_NAME}-access"
    ]
    
    for policy_name in policies:
        try:
            # Determine policy type
            if 'encryption' in policy_name:
                policy_type = 'encryption'
            elif 'network' in policy_name:
                policy_type = 'network'
            else:
                policy_type = 'data'
            
            if policy_type == 'data':
                aoss_client.delete_access_policy(name=policy_name, type=policy_type)
            else:
                aoss_client.delete_security_policy(name=policy_name, type=policy_type)
            
            print(f"✓ Deleted policy: {policy_name}")
        except ClientError as e:
            print(f"  Policy deletion ({policy_name}): {e}")

def delete_s3_bucket():
    """Delete S3 bucket and all contents"""
    try:
        # Delete all objects first
        response = s3_client.list_objects_v2(Bucket=BUCKET_NAME)
        
        if 'Contents' in response:
            objects = [{'Key': obj['Key']} for obj in response['Contents']]
            s3_client.delete_objects(Bucket=BUCKET_NAME, Delete={'Objects': objects})
            print(f"✓ Deleted {len(objects)} objects from bucket")
        
        # Delete bucket
        s3_client.delete_bucket(Bucket=BUCKET_NAME)
        print(f"✓ Deleted S3 bucket: {BUCKET_NAME}")
        
    except ClientError as e:
        print(f"  Bucket deletion: {e}")

def delete_iam_roles():
    """Delete IAM roles"""
    roles = [
        'BedrockExecutionRoleForKnowledgeBase',
        'AmazonBedrockExecutionRoleForAgents',
        'EcommerceLambdaExecutionRole'
    ]
    
    for role_name in roles:
        try:
            # Detach managed policies
            response = iam_client.list_attached_role_policies(RoleName=role_name)
            for policy in response.get('AttachedPolicies', []):
                iam_client.detach_role_policy(
                    RoleName=role_name,
                    PolicyArn=policy['PolicyArn']
                )
            
            # Delete inline policies
            response = iam_client.list_role_policies(RoleName=role_name)
            for policy_name in response.get('PolicyNames', []):
                iam_client.delete_role_policy(
                    RoleName=role_name,
                    PolicyName=policy_name
                )
            
            # Delete role
            iam_client.delete_role(RoleName=role_name)
            print(f"✓ Deleted IAM role: {role_name}")
            
        except ClientError as e:
            print(f"  Role deletion ({role_name}): {e}")

if __name__ == "__main__":
    print("=" * 60)
    print("CLEANUP: Deleting All E-commerce Agent Resources")
    print("=" * 60)
    print("\nWARNING: This will delete all resources and cannot be undone!")
    
    confirm = input("\nType 'DELETE' to confirm: ")
    
    if confirm != 'DELETE':
        print("Cleanup cancelled.")
        exit(0)
    
    print("\nStarting cleanup...")
    
    # Delete in order (dependencies first)
    delete_agent()
    time.sleep(2)
    
    delete_knowledge_base()
    time.sleep(2)
    
    delete_guardrail()
    time.sleep(2)
    
    delete_lambda_functions()
    time.sleep(2)
    
    delete_opensearch_collection()
    time.sleep(2)
    
    delete_s3_bucket()
    time.sleep(2)
    
    delete_iam_roles()
    
    print("\n" + "=" * 60)
    print("✓ Cleanup complete!")
    print("=" * 60)
    print("\nAll resources have been deleted.")
    print("Please verify in AWS Console:")
    print("  - Bedrock → Agents")
    print("  - Bedrock → Knowledge Bases")
    print("  - Bedrock → Guardrails")
    print("  - Lambda → Functions")
    print("  - OpenSearch → Collections")
    print("  - S3 → Buckets")
    print("  - IAM → Roles")
```

**Run cleanup:**
```bash
python notebooks/13_cleanup_all.py
```

---

## Next Steps

### 1. Production Deployment

- **Create Agent Alias**: Deploy stable version
- **Set up Monitoring**: CloudWatch metrics and alarms
- **Implement Logging**: Comprehensive request/response logging
- **Add Analytics**: Track user queries and satisfaction

### 2. Advanced Features

- **Multi-language Support**: Translate queries and responses
- **Voice Integration**: Amazon Lex or custom STT/TTS
- **Image Search**: Visual product discovery
- **Recommendation Engine**: Personalized product suggestions
- **A/B Testing**: Test different prompts and configurations

### 3. Integration

- **Web Application**: React/Vue frontend
- **Mobile App**: iOS/Android native apps
- **Chat Widget**: Embed in website
- **Messaging Platforms**: WhatsApp, Facebook Messenger
- **Voice Assistants**: Alexa, Google Home

### 4. Optimization

- **Response Caching**: Cache frequent queries
- **Query Preprocessing**: Clean and normalize input
- **Contextual Routing**: Route to specialized sub-agents
- **Load Testing**: Measure throughput and latency
- **Cost Optimization**: Monitor and reduce API calls

### 5. Learning Resources

**AWS Documentation:**
- [Bedrock Agents](https://docs.aws.amazon.com/bedrock/latest/userguide/agents.html)
- [Knowledge Bases](https://docs.aws.amazon.com/bedrock/latest/userguide/knowledge-base.html)
- [Guardrails](https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails.html)

**LangChain:**
- [LangChain Docs](https://python.langchain.com/docs/get_started/introduction)
- [LangChain AWS](https://python.langchain.com/docs/integrations/platforms/aws)

**Community:**
- AWS Bedrock Community Forum
- LangChain Discord
- GitHub Issues and Discussions

---

## Conclusion

Congratulations! You've built an enterprise-grade e-commerce AI agent with:

✓ **Foundation Model**: Claude 3.5 Sonnet for intelligent responses  
✓ **RAG System**: Knowledge Base with product catalog and policies  
✓ **Agent Framework**: Bedrock Agents for orchestration  
✓ **Safety Controls**: Guardrails for content filtering and compliance  
✓ **Action Groups**: Lambda functions for inventory and orders  
✓ **LangChain Integration**: Advanced workflow orchestration  
✓ **Production-Ready**: Comprehensive testing and error handling

Your agent can:
- Answer product questions using RAG
- Check inventory in real-time
- Track orders and shipments
- Maintain conversation context
- Block inappropriate content
- Handle complex multi-turn conversations

### Key Takeaways

1. **RAG is Powerful**: Provides accurate, up-to-date information without retraining
2. **Guardrails are Essential**: Safety and compliance are non-negotiable
3. **Agent Architecture**: Orchestrates multiple components seamlessly
4. **Testing is Critical**: Comprehensive tests ensure production readiness
5. **LangChain Adds Value**: Workflow management and observability

---

## Support

**Questions or Issues?**
- Create an issue in the repository
- Check AWS Documentation
- Join the community forums

**Contributing:**
- Pull requests welcome
- Share your improvements
- Report bugs and suggest features

---

**Last Updated:** February 2026  
**Version:** 1.0  
**Author:** Your Organization  
**License:** MIT

---
