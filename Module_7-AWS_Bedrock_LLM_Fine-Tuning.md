# AWS Bedrock LLM Fine-Tuning: Complete Guide

**Estimated Time:** 90-120 minutes  
**Difficulty Level:** Advanced  
**Last Updated:** February 2026

## Table of Contents
1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Understanding Fine-Tuning in Bedrock](#understanding-fine-tuning-in-bedrock)
4. [Supported Models for Fine-Tuning](#supported-models-for-fine-tuning)
5. [Preparing Fine-Tuning Datasets](#preparing-fine-tuning-datasets)
6. [Step-by-Step: Creating a Fine-Tuning Job](#step-by-step-creating-a-fine-tuning-job)
7. [Prompt Engineering for Model Specialization](#prompt-engineering-for-model-specialization)
8. [Evaluation Metrics and Benchmarks](#evaluation-metrics-and-benchmarks)
9. [Monitoring and Testing Your Fine-Tuned Model](#monitoring-and-testing-your-fine-tuned-model)
10. [Troubleshooting](#troubleshooting)
11. [Cleanup](#cleanup)
12. [Cost Considerations](#cost-considerations)
13. [Next Steps](#next-steps)

---

## Overview

AWS Bedrock's fine-tuning capability allows you to customize foundation models (FMs) with your own data to improve performance on domain-specific tasks. Unlike prompt engineering, which provides instructions at inference time, fine-tuning creates a new custom model by training the base model on your labeled examples.

### What is Fine-Tuning?

Fine-tuning is a machine learning technique where you take a pre-trained foundation model and continue training it on your specific dataset. This process:
- Adapts the model to your domain-specific language, terminology, and patterns
- Improves performance on specialized tasks
- Reduces the need for lengthy prompts at inference time
- Creates a dedicated model version for your use case

### When to Use Fine-Tuning vs. Prompt Engineering

| Approach | Best For | Considerations |
|----------|----------|----------------|
| **Prompt Engineering** | Quick iterations, changing requirements, general tasks | No additional cost, faster to implement |
| **RAG (Retrieval-Augmented Generation)** | Knowledge-intensive tasks, dynamic information | Good for facts and references |
| **Fine-Tuning** | Consistent format/style, domain terminology, specialized behavior | Requires training data, cost, and time |

---

## Prerequisites

### AWS Account Setup
- AWS account with appropriate permissions
- IAM role with fine-tuning permissions
- S3 bucket for training data storage
- Sufficient service quotas for fine-tuning jobs

### Required IAM Permissions

Create an IAM policy with the following permissions:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "bedrock:CreateModelCustomizationJob",
                "bedrock:GetModelCustomizationJob",
                "bedrock:ListModelCustomizationJobs",
                "bedrock:StopModelCustomizationJob",
                "bedrock:DeleteCustomModel",
                "bedrock:GetCustomModel",
                "bedrock:ListCustomModels",
                "bedrock:CreateProvisionedModelThroughput",
                "bedrock:GetProvisionedModelThroughput",
                "bedrock:DeleteProvisionedModelThroughput",
                "bedrock:InvokeModel"
            ],
            "Resource": "*"
        },
        {
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:PutObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::your-training-bucket/*",
                "arn:aws:s3:::your-training-bucket"
            ]
        },
        {
            "Effect": "Allow",
            "Action": [
                "iam:PassRole"
            ],
            "Resource": "arn:aws:iam::*:role/BedrockFineTuningRole"
        }
    ]
}
```

### Local Development Environment
- Python 3.8 or higher
- AWS CLI installed and configured
- boto3 library

**Installation Commands:**

```bash
# Bash/Linux/macOS
pip install boto3 awscli pandas jsonlines

# Verify installation
aws --version
python -c "import boto3; print(boto3.__version__)"
```

```powershell
# PowerShell/Windows
pip install boto3 awscli pandas jsonlines

# Verify installation
aws --version
python -c "import boto3; print(boto3.__version__)"
```

### Enable Model Access

1. Navigate to AWS Console → Bedrock → **Model catalog**
2. Select the base model you want to fine-tune (e.g., Claude 3 Haiku)
3. Click **Request model access** if not already enabled
4. Wait for access approval (usually instant for Claude models)

---

## Understanding Fine-Tuning in Bedrock

### Fine-Tuning Architecture

```
┌─────────────────┐
│  Base Model     │
│  (Foundation)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Your Training   │◄──── Custom Dataset (JSONL)
│     Data        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Fine-Tuning    │
│     Process     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Custom Model    │◄──── Your specialized model
│   (Fine-Tuned)  │
└─────────────────┘
```

### Key Concepts

1. **Base Model**: The pre-trained foundation model (e.g., Claude 3 Haiku, Titan Text)
2. **Training Data**: Your labeled examples in JSONL format
3. **Validation Data**: Separate dataset to evaluate model performance
4. **Hyperparameters**: Training configuration (epochs, learning rate, batch size)
5. **Custom Model**: The resulting fine-tuned model
6. **Provisioned Throughput**: Dedicated capacity for inference (required for some models)

---

## Supported Models for Fine-Tuning

### Currently Supported Models (as of February 2026)

| Model Family | Model ID | Fine-Tuning Support | Min Training Examples | Max Training Examples |
|-------------|----------|---------------------|----------------------|----------------------|
| **Amazon Titan Text** | `amazon.titan-text-express-v1` | ✅ | 32 | 10,000 |
| **Amazon Titan Text** | `amazon.titan-text-lite-v1` | ✅ | 32 | 10,000 |
| **Anthropic Claude** | `anthropic.claude-3-haiku-20240307-v1:0` | ✅ | 32 | 10,000 |
| **Cohere Command** | `cohere.command-light-text-v14` | ✅ | 32 | 10,000 |
| **Meta Llama** | `meta.llama2-13b-v1` | ✅ | 32 | 10,000 |

### Model Selection Criteria

**Amazon Titan Text Express** - Best for:
- General-purpose text generation
- Cost-effective fine-tuning
- Enterprise applications with moderate complexity

**Claude 3 Haiku** - Best for:
- Conversational AI applications
- High-quality reasoning tasks
- Complex instruction following
- Customer service and support

**Cohere Command** - Best for:
- Multilingual applications
- Classification tasks
- Text summarization

**Meta Llama 2** - Best for:
- Open-source model requirements
- Research and development
- Custom deployments

### Checking Model Availability

```python
import boto3

bedrock = boto3.client('bedrock', region_name='us-east-1')

# List available foundation models
response = bedrock.list_foundation_models(
    byCustomizationType='FINE_TUNING'
)

print("Models supporting fine-tuning:")
for model in response['modelSummaries']:
    print(f"- {model['modelId']}")
    print(f"  Name: {model['modelName']}")
    print(f"  Provider: {model['providerName']}")
    print(f"  Customizations: {model['customizationsSupported']}")
    print()
```

---

## Preparing Fine-Tuning Datasets

### Dataset Format Requirements

Bedrock requires training data in **JSONL (JSON Lines)** format, where each line is a valid JSON object representing a single training example.

### Format Structure

**Prompt-Completion Format** (Most Common):
```jsonl
{"prompt": "User input or question", "completion": "Expected model response"}
{"prompt": "Another user input", "completion": "Another expected response"}
```

**Chat Format** (For conversational models):
```jsonl
{"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there!"}]}
{"messages": [{"role": "user", "content": "What is AI?"}, {"role": "assistant", "content": "AI stands for Artificial Intelligence..."}]}
```

### Real-World Example: Customer Support Dataset

Let's create a dataset for fine-tuning a customer support model for a fictional e-commerce company:

**customer_support_training.jsonl**
```jsonl
{"prompt": "How do I track my order?", "completion": "To track your order, please log into your account at shop.example.com and navigate to 'My Orders'. Click on the order number to view real-time tracking information. You'll also receive tracking updates via email."}
{"prompt": "What's your return policy?", "completion": "We offer a 30-day return policy on all items. Products must be unused and in original packaging. To initiate a return, visit your account dashboard and select 'Return Item' next to your order. You'll receive a prepaid shipping label via email."}
{"prompt": "My package arrived damaged, what should I do?", "completion": "I'm sorry to hear your package arrived damaged. Please take photos of the damaged item and packaging, then submit a damage claim through your account dashboard under 'Order Issues'. We'll process a replacement or refund within 24 hours."}
{"prompt": "Can I change my shipping address?", "completion": "If your order hasn't shipped yet, you can update the shipping address in your account under 'My Orders' > 'Edit Order'. If the order has already shipped, please contact our support team at support@example.com immediately for assistance."}
{"prompt": "Do you offer international shipping?", "completion": "Yes, we ship to over 50 countries worldwide. International shipping costs and delivery times vary by destination. You can see available shipping options and costs during checkout. Note that customs fees may apply based on your country's regulations."}
```

### Dataset Quality Guidelines

#### 1. **Minimum Requirements**
- **Minimum examples**: 32 (but 100-500 recommended for good results)
- **Maximum examples**: 10,000
- **Validation split**: 10-20% of total data

#### 2. **Quality Characteristics**

**✅ Good Training Examples:**
- Consistent format across all examples
- Natural, varied language (avoid repetition)
- Representative of actual use cases
- Accurate and complete responses
- Appropriate length (not too short or verbose)

**❌ Poor Training Examples:**
- Inconsistent formatting
- Repetitive or template-like text
- Factually incorrect information
- Overly short or overly long responses
- Examples that don't match real usage

#### 3. **Diversity and Balance**
- Include various question types
- Cover different scenarios and edge cases
- Balance representation of different topics
- Include different response lengths
- Mix simple and complex examples

### Creating a Dataset: Step-by-Step

**Step 1: Collect Raw Data**

Create a Python script to structure your data:

```python
import json
import random

# Example: Converting CSV to JSONL
def create_training_dataset():
    training_examples = [
        {
            "prompt": "How do I reset my password?",
            "completion": "To reset your password, click 'Forgot Password' on the login page. Enter your email address, and we'll send you a reset link. The link expires in 24 hours for security."
        },
        {
            "prompt": "What payment methods do you accept?",
            "completion": "We accept all major credit cards (Visa, MasterCard, American Express, Discover), PayPal, Apple Pay, and Google Pay. For large orders, we also offer bank transfers and purchase orders for business customers."
        },
        # Add more examples...
    ]
    
    return training_examples

def save_jsonl(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

# Create dataset
data = create_training_dataset()

# Shuffle data
random.shuffle(data)

# Split into training and validation (80/20 split)
split_idx = int(len(data) * 0.8)
training_data = data[:split_idx]
validation_data = data[split_idx:]

# Save files
save_jsonl(training_data, 'training_data.jsonl')
save_jsonl(validation_data, 'validation_data.jsonl')

print(f"Created training set with {len(training_data)} examples")
print(f"Created validation set with {len(validation_data)} examples")
```

**Step 2: Validate Dataset Format**

```python
import jsonlines

def validate_jsonl_file(filepath):
    """Validate JSONL file format and content"""
    errors = []
    line_count = 0
    
    try:
        with jsonlines.open(filepath) as reader:
            for idx, line in enumerate(reader, 1):
                line_count += 1
                
                # Check required fields
                if 'prompt' not in line or 'completion' not in line:
                    errors.append(f"Line {idx}: Missing 'prompt' or 'completion' field")
                
                # Check for empty values
                if not line.get('prompt', '').strip():
                    errors.append(f"Line {idx}: Empty prompt")
                if not line.get('completion', '').strip():
                    errors.append(f"Line {idx}: Empty completion")
                
                # Check length (adjust limits based on your needs)
                if len(line.get('prompt', '')) > 2048:
                    errors.append(f"Line {idx}: Prompt exceeds 2048 characters")
                if len(line.get('completion', '')) > 2048:
                    errors.append(f"Line {idx}: Completion exceeds 2048 characters")
    
    except Exception as e:
        errors.append(f"File parsing error: {str(e)}")
    
    print(f"Validation Results for {filepath}:")
    print(f"- Total lines: {line_count}")
    print(f"- Errors found: {len(errors)}")
    
    if errors:
        print("\nErrors:")
        for error in errors[:10]:  # Show first 10 errors
            print(f"  {error}")
    else:
        print("✅ File is valid!")
    
    return len(errors) == 0

# Validate your files
validate_jsonl_file('training_data.jsonl')
validate_jsonl_file('validation_data.jsonl')
```

**Step 3: Upload to S3**

```bash
# Bash/Linux/macOS
aws s3 mb s3://my-bedrock-finetuning-bucket
aws s3 cp training_data.jsonl s3://my-bedrock-finetuning-bucket/datasets/
aws s3 cp validation_data.jsonl s3://my-bedrock-finetuning-bucket/datasets/

# Verify upload
aws s3 ls s3://my-bedrock-finetuning-bucket/datasets/
```

```powershell
# PowerShell/Windows
aws s3 mb s3://my-bedrock-finetuning-bucket
aws s3 cp training_data.jsonl s3://my-bedrock-finetuning-bucket/datasets/
aws s3 cp validation_data.jsonl s3://my-bedrock-finetuning-bucket/datasets/

# Verify upload
aws s3 ls s3://my-bedrock-finetuning-bucket/datasets/
```

---

## Step-by-Step: Creating a Fine-Tuning Job

### Step 1: Create IAM Role for Bedrock

Bedrock needs permissions to access your S3 bucket and create the custom model.

**Create trust policy** (`bedrock-trust-policy.json`):
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "Service": "bedrock.amazonaws.com"
            },
            "Action": "sts:AssumeRole"
        }
    ]
}
```

**Create permissions policy** (`bedrock-permissions-policy.json`):
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:PutObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::my-bedrock-finetuning-bucket",
                "arn:aws:s3:::my-bedrock-finetuning-bucket/*"
            ]
        }
    ]
}
```

**Create the role:**

```bash
# Bash/Linux/macOS
# Create the role
aws iam create-role \
    --role-name BedrockFineTuningRole \
    --assume-role-policy-document file://bedrock-trust-policy.json

# Attach the permissions policy
aws iam put-role-policy \
    --role-name BedrockFineTuningRole \
    --policy-name BedrockS3Access \
    --policy-document file://bedrock-permissions-policy.json

# Get the role ARN (save this for later)
aws iam get-role --role-name BedrockFineTuningRole --query 'Role.Arn' --output text
```

```powershell
# PowerShell/Windows
# Create the role
aws iam create-role `
    --role-name BedrockFineTuningRole `
    --assume-role-policy-document file://bedrock-trust-policy.json

# Attach the permissions policy
aws iam put-role-policy `
    --role-name BedrockFineTuningRole `
    --policy-name BedrockS3Access `
    --policy-document file://bedrock-permissions-policy.json

# Get the role ARN (save this for later)
aws iam get-role --role-name BedrockFineTuningRole --query 'Role.Arn' --output text
```

### Step 2: Create Fine-Tuning Job (Python SDK)

```python
import boto3
import time
from datetime import datetime

# Initialize Bedrock client
bedrock = boto3.client('bedrock', region_name='us-east-1')

# Configuration
ROLE_ARN = "arn:aws:iam::YOUR_ACCOUNT_ID:role/BedrockFineTuningRole"  # Replace with your role ARN
BASE_MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0"
TRAINING_DATA_URI = "s3://my-bedrock-finetuning-bucket/datasets/training_data.jsonl"
VALIDATION_DATA_URI = "s3://my-bedrock-finetuning-bucket/datasets/validation_data.jsonl"
OUTPUT_DATA_URI = "s3://my-bedrock-finetuning-bucket/output/"
CUSTOM_MODEL_NAME = f"customer-support-model-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

def create_finetuning_job():
    """Create a fine-tuning job in Bedrock"""
    
    try:
        response = bedrock.create_model_customization_job(
            jobName=CUSTOM_MODEL_NAME,
            customModelName=CUSTOM_MODEL_NAME,
            roleArn=ROLE_ARN,
            baseModelIdentifier=BASE_MODEL_ID,
            trainingDataConfig={
                's3Uri': TRAINING_DATA_URI
            },
            validationDataConfig={
                's3Uri': VALIDATION_DATA_URI
            },
            outputDataConfig={
                's3Uri': OUTPUT_DATA_URI
            },
            hyperParameters={
                'epochCount': '3',
                'batchSize': '1',
                'learningRate': '0.00001'
            }
        )
        
        job_arn = response['jobArn']
        print(f"✅ Fine-tuning job created successfully!")
        print(f"Job ARN: {job_arn}")
        print(f"Custom Model Name: {CUSTOM_MODEL_NAME}")
        
        return job_arn
    
    except Exception as e:
        print(f"❌ Error creating fine-tuning job: {str(e)}")
        return None

def monitor_finetuning_job(job_arn):
    """Monitor the progress of a fine-tuning job"""
    
    print("\n📊 Monitoring fine-tuning job progress...")
    print("This may take 2-8 hours depending on dataset size and model.\n")
    
    while True:
        try:
            response = bedrock.get_model_customization_job(
                jobIdentifier=job_arn
            )
            
            status = response['status']
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Status: {status}")
            
            if status == 'Completed':
                print("\n✅ Fine-tuning completed successfully!")
                print(f"Custom Model ARN: {response['outputModelArn']}")
                print(f"Custom Model Name: {response['customModelName']}")
                return response
            
            elif status == 'Failed':
                print("\n❌ Fine-tuning job failed!")
                if 'failureMessage' in response:
                    print(f"Failure reason: {response['failureMessage']}")
                return None
            
            elif status in ['Stopped', 'Stopping']:
                print("\n⚠️ Fine-tuning job was stopped")
                return None
            
            # Wait 5 minutes before checking again
            time.sleep(300)
        
        except KeyboardInterrupt:
            print("\n⚠️ Monitoring interrupted. Job continues in background.")
            print(f"Check status with: aws bedrock get-model-customization-job --job-identifier {job_arn}")
            break
        except Exception as e:
            print(f"Error checking job status: {str(e)}")
            time.sleep(60)

# Execute fine-tuning
if __name__ == "__main__":
    print("🚀 Starting fine-tuning process...\n")
    
    # Create the job
    job_arn = create_finetuning_job()
    
    if job_arn:
        # Monitor until completion
        result = monitor_finetuning_job(job_arn)
        
        if result:
            print("\n📋 Final Job Details:")
            print(f"- Model Name: {result['customModelName']}")
            print(f"- Model ARN: {result['outputModelArn']}")
            print(f"- Training Loss: {result.get('trainingMetrics', {}).get('trainingLoss', 'N/A')}")
```

### Step 3: Create Fine-Tuning Job (AWS Console)

**Alternative: Using AWS Console**

1. Navigate to **AWS Console** → **Bedrock** → **Custom models**
2. Click **Create customization job**
3. Configure the job:
   - **Job name**: `customer-support-model`
   - **Base model**: Select `Claude 3 Haiku` or your preferred model
   - **Training data**: Browse to your S3 training data URI
   - **Validation data**: Browse to your S3 validation data URI
   - **Output location**: Specify S3 output path
   - **IAM role**: Select `BedrockFineTuningRole`
4. Configure **Hyperparameters**:
   - **Epochs**: 3 (default, increase for larger datasets)
   - **Batch size**: 1 (default)
   - **Learning rate**: 0.00001 (default)
5. Click **Create customization job**

### Step 4: Understanding Hyperparameters

| Hyperparameter | Description | Typical Values | Impact |
|----------------|-------------|----------------|--------|
| **Epochs** | Number of complete passes through training data | 1-10 | More epochs = better learning but risk overfitting |
| **Batch Size** | Number of examples processed together | 1-8 | Higher = faster but more memory |
| **Learning Rate** | How quickly model weights are updated | 0.000001-0.0001 | Lower = slower but more stable |

**Recommendations:**
- **Small datasets (<500 examples)**: 3-5 epochs
- **Medium datasets (500-2000)**: 2-3 epochs
- **Large datasets (>2000)**: 1-2 epochs

---

## Prompt Engineering for Model Specialization

### Pre-Fine-Tuning vs. Post-Fine-Tuning Prompts

Fine-tuning changes how you should structure prompts:

**Before Fine-Tuning:**
```python
prompt = """You are a customer support agent for Example Shop. 
Follow these guidelines:
1. Be polite and professional
2. Reference our 30-day return policy
3. Direct users to shop.example.com for account issues
4. Offer proactive solutions

Question: How do I track my order?

Answer:"""
```

**After Fine-Tuning:**
```python
# Much simpler - the model learned the context
prompt = "How do I track my order?"
```

### Designing Effective Training Prompts

#### 1. **Consistent Format Pattern**

Choose ONE format and use it consistently across all training examples:

**Option A: Direct Q&A**
```jsonl
{"prompt": "What is your return policy?", "completion": "We offer..."}
```

**Option B: Instructional**
```jsonl
{"prompt": "Answer this customer question: What is your return policy?", "completion": "We offer..."}
```

**Option C: Role-based**
```jsonl
{"prompt": "Customer: What is your return policy?\nAgent:", "completion": "We offer..."}
```

#### 2. **Include Context in Training Data**

For specialized behavior, embed instructions in your training examples:

```jsonl
{"prompt": "A frustrated customer asks: My order is late, where is it?!", "completion": "I sincerely apologize for the delay in your order. I understand how frustrating this must be. Let me help you track it down right away. Could you please provide your order number so I can investigate immediately?"}
{"prompt": "A customer urgently asks: I need to cancel my order NOW!", "completion": "I can help you with that right away. If your order hasn't shipped yet, I can cancel it immediately. Please provide your order number, and I'll check the current status for you."}
```

#### 3. **Edge Case Handling**

Train the model on edge cases and special scenarios:

```jsonl
{"prompt": "Can I return an item after 45 days?", "completion": "Our standard return policy is 30 days from delivery. However, I'd like to help you. Could you tell me more about the situation? In some cases, we may be able to make an exception or offer store credit."}
{"prompt": "I want to speak to a human, not a bot!", "completion": "I understand, and I'm here to help connect you with our team. Let me get some basic information first so the specialist can assist you efficiently. What's your order number and the main issue you're experiencing?"}
{"prompt": "Your service is terrible!", "completion": "I'm truly sorry you've had a negative experience. Your feedback is important to us. I want to make this right. Could you please tell me what happened so I can help resolve the issue immediately?"}
```

### Specialization Techniques

#### Technique 1: Domain-Specific Terminology

Train the model to use your company's specific terms:

```jsonl
{"prompt": "What is Premium Plus membership?", "completion": "Premium Plus is our top-tier membership that includes free express shipping, early access to sales, 24/7 priority support, and exclusive member discounts. It costs $99/year and typically pays for itself after 3-4 orders."}
{"prompt": "How does Express Plus shipping work?", "completion": "Express Plus is our fastest shipping option - guaranteed delivery within 24 hours for orders placed before 2 PM. It's available in most major cities and costs $15.99 per order, or it's free for Premium Plus members."}
```

#### Technique 2: Tone and Style Consistency

```jsonl
{"prompt": "Do you have customer support?", "completion": "Absolutely! Our support team is available 24/7 via live chat, email, or phone. We typically respond to live chat within 2 minutes and emails within 4 hours. How can we help you today?"}
{"prompt": "Are you open on weekends?", "completion": "Yes, we're here to help 7 days a week! Our customer service team is available 24/7, and our warehouse processes orders every day including weekends. Saturday and Sunday orders ship the same day if placed before 2 PM."}
```

#### Technique 3: Multi-Step Reasoning

Train the model to handle complex, multi-step processes:

```jsonl
{"prompt": "I received the wrong item and I need it by Friday for a gift. What should I do?", "completion": "I'm so sorry about the mix-up, and I understand the urgency for your gift! Here's how we'll fix this: 1) I'll immediately process a replacement order with express shipping at no charge - you'll receive it by Thursday. 2) For the incorrect item, I'll email you a prepaid return label - no rush on sending it back. 3) I'll also apply a 15% courtesy discount to your account for the inconvenience. Sound good?"}
```

---

## Evaluation Metrics and Benchmarks

### Understanding Model Performance

After fine-tuning completes, Bedrock provides several metrics to evaluate your custom model:

### 1. Training Metrics

**Training Loss**
- Measures how well the model fits the training data
- Lower values = better fit to training examples
- Formula: Cross-entropy loss between predicted and actual completions

```python
def interpret_training_loss(loss):
    """Interpret training loss values"""
    if loss < 0.5:
        return "Excellent - Model learned training data very well"
    elif loss < 1.0:
        return "Good - Model learned training data well"
    elif loss < 2.0:
        return "Fair - Model learned basic patterns"
    else:
        return "Poor - Consider more epochs or better data quality"

# Example from job results
training_loss = 0.45
print(f"Training Loss: {training_loss}")
print(f"Interpretation: {interpret_training_loss(training_loss)}")
```

**Validation Loss**
- Measures performance on unseen validation data
- Indicates generalization ability
- Should be similar to training loss

```python
def check_overfitting(train_loss, val_loss):
    """Check if model is overfitting"""
    difference = val_loss - train_loss
    
    if difference < 0.2:
        return "✅ Good generalization - Model will perform well on new data"
    elif difference < 0.5:
        return "⚠️ Slight overfitting - Model may perform worse on new data"
    else:
        return "❌ Significant overfitting - Reduce epochs or increase training data"

# Example
print(check_overfitting(train_loss=0.45, val_loss=0.52))
```

### 2. Retrieving Metrics from Your Job

```python
import boto3

bedrock = boto3.client('bedrock', region_name='us-east-1')

def get_job_metrics(job_arn):
    """Retrieve and display metrics from fine-tuning job"""
    
    response = bedrock.get_model_customization_job(
        jobIdentifier=job_arn
    )
    
    print("📊 Fine-Tuning Metrics Report")
    print("=" * 50)
    
    # Training metrics
    if 'trainingMetrics' in response:
        metrics = response['trainingMetrics']
        print("\n🎯 Training Performance:")
        print(f"  - Training Loss: {metrics.get('trainingLoss', 'N/A')}")
        
    # Validation metrics
    if 'validationMetrics' in response:
        val_metrics = response['validationMetrics']
        print("\n✅ Validation Performance:")
        for metric in val_metrics:
            print(f"  - {metric['name']}: {metric['value']}")
    
    # Output location
    if 'outputDataConfig' in response:
        print(f"\n📁 Output Location:")
        print(f"  {response['outputDataConfig']['s3Uri']}")
    
    return response

# Usage
job_arn = "arn:aws:bedrock:us-east-1:123456789012:model-customization-job/xyz"
metrics = get_job_metrics(job_arn)
```

### 3. Custom Evaluation Framework

Create your own evaluation suite to test model quality:

```python
import boto3
import json
from typing import List, Dict

bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-east-1')

class ModelEvaluator:
    def __init__(self, base_model_id: str, custom_model_arn: str):
        self.base_model_id = base_model_id
        self.custom_model_arn = custom_model_arn
        self.results = []
    
    def invoke_model(self, model_id: str, prompt: str) -> str:
        """Invoke a model and return the response"""
        
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 500,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        })
        
        response = bedrock_runtime.invoke_model(
            modelId=model_id,
            body=body
        )
        
        response_body = json.loads(response['body'].read())
        return response_body['content'][0]['text']
    
    def evaluate_test_case(self, prompt: str, expected_keywords: List[str]):
        """Compare base model vs fine-tuned model"""
        
        print(f"\n📝 Testing: {prompt[:50]}...")
        
        # Get responses from both models
        base_response = self.invoke_model(self.base_model_id, prompt)
        custom_response = self.invoke_model(self.custom_model_arn, prompt)
        
        # Score based on keyword presence
        base_score = sum(1 for kw in expected_keywords if kw.lower() in base_response.lower())
        custom_score = sum(1 for kw in expected_keywords if kw.lower() in custom_response.lower())
        
        result = {
            'prompt': prompt,
            'base_model_score': base_score,
            'custom_model_score': custom_score,
            'improvement': custom_score - base_score,
            'expected_keywords': expected_keywords
        }
        
        self.results.append(result)
        
        print(f"  Base Model Score: {base_score}/{len(expected_keywords)}")
        print(f"  Custom Model Score: {custom_score}/{len(expected_keywords)}")
        print(f"  Improvement: {'+' if result['improvement'] >= 0 else ''}{result['improvement']}")
        
        return result
    
    def run_evaluation(self, test_cases: List[Dict]):
        """Run full evaluation suite"""
        
        print("🧪 Starting Model Evaluation")
        print("=" * 60)
        
        for test in test_cases:
            self.evaluate_test_case(
                prompt=test['prompt'],
                expected_keywords=test['expected_keywords']
            )
        
        self.print_summary()
    
    def print_summary(self):
        """Print evaluation summary"""
        
        print("\n" + "=" * 60)
        print("📊 EVALUATION SUMMARY")
        print("=" * 60)
        
        total_tests = len(self.results)
        improved = sum(1 for r in self.results if r['improvement'] > 0)
        same = sum(1 for r in self.results if r['improvement'] == 0)
        worse = sum(1 for r in self.results if r['improvement'] < 0)
        
        avg_improvement = sum(r['improvement'] for r in self.results) / total_tests
        
        print(f"\nTotal Tests: {total_tests}")
        print(f"Improved: {improved} ({improved/total_tests*100:.1f}%)")
        print(f"Same: {same} ({same/total_tests*100:.1f}%)")
        print(f"Worse: {worse} ({worse/total_tests*100:.1f}%)")
        print(f"Average Improvement: {avg_improvement:.2f} keywords")
        
        if improved > worse:
            print("\n✅ Fine-tuned model shows clear improvement!")
        elif improved == worse:
            print("\n⚠️ Mixed results - consider more training data or epochs")
        else:
            print("\n❌ Fine-tuned model underperforms - review training data quality")

# Example usage
evaluator = ModelEvaluator(
    base_model_id="anthropic.claude-3-haiku-20240307-v1:0",
    custom_model_arn="arn:aws:bedrock:us-east-1:123456789012:custom-model/xyz"
)

# Define test cases
test_cases = [
    {
        'prompt': 'How do I track my order?',
        'expected_keywords': ['log into', 'account', 'My Orders', 'tracking', 'email']
    },
    {
        'prompt': 'What is your return policy?',
        'expected_keywords': ['30-day', 'unused', 'original packaging', 'dashboard', 'prepaid']
    },
    {
        'prompt': 'Do you ship internationally?',
        'expected_keywords': ['50 countries', 'checkout', 'customs', 'delivery times']
    },
    {
        'prompt': 'My package is damaged',
        'expected_keywords': ['photos', 'damage claim', 'dashboard', '24 hours', 'replacement']
    }
]

# Run evaluation
evaluator.run_evaluation(test_cases)
```

### 4. Human Evaluation Metrics

Quantitative metrics don't tell the whole story. Create a human evaluation framework:

**Evaluation Rubric:**

| Criteria | Score 1 (Poor) | Score 3 (Good) | Score 5 (Excellent) |
|----------|---------------|----------------|---------------------|
| **Accuracy** | Incorrect info | Mostly correct | Completely accurate |
| **Relevance** | Off-topic | Somewhat relevant | Directly addresses question |
| **Tone** | Inappropriate | Acceptable | Perfect for context |
| **Completeness** | Missing key info | Adequate | Comprehensive |
| **Formatting** | Confusing | Readable | Well-structured |

**Evaluation Template:**

```python
def human_evaluation_form():
    """Template for human evaluators"""
    
    evaluation = {
        'test_id': '',
        'prompt': '',
        'model_response': '',
        'scores': {
            'accuracy': 0,      # 1-5
            'relevance': 0,     # 1-5
            'tone': 0,          # 1-5
            'completeness': 0,  # 1-5
            'formatting': 0     # 1-5
        },
        'notes': '',
        'would_use_in_production': False  # Yes/No
    }
    
    return evaluation
```

### 5. A/B Testing in Production

Once deployed, run A/B tests to compare models:

```python
import random

def route_to_model(user_id: str):
    """Route 50% of traffic to each model"""
    
    # Consistent routing based on user ID
    if hash(user_id) % 2 == 0:
        return {
            'model': 'base_model',
            'model_id': 'anthropic.claude-3-haiku-20240307-v1:0'
        }
    else:
        return {
            'model': 'fine_tuned_model',
            'model_id': 'arn:aws:bedrock:us-east-1:123456789012:custom-model/xyz'
        }

# Track metrics
def log_interaction(user_id, model, prompt, response, feedback):
    """Log interaction for analysis"""
    
    metrics = {
        'timestamp': datetime.now().isoformat(),
        'user_id': user_id,
        'model': model,
        'prompt': prompt,
        'response': response,
        'user_feedback': feedback,  # thumbs up/down
        'response_time_ms': 0
    }
    
    # Send to your analytics system
    # e.g., CloudWatch, Kinesis, or custom database
```

---

## Monitoring and Testing Your Fine-Tuned Model

### Deploying Your Custom Model

**Option 1: On-Demand Inference** (Available for some models)

```python
import boto3
import json

bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-east-1')

def invoke_custom_model(custom_model_arn: str, prompt: str):
    """Invoke your fine-tuned model"""
    
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1000,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.7
    })
    
    try:
        response = bedrock_runtime.invoke_model(
            modelId=custom_model_arn,
            body=body
        )
        
        response_body = json.loads(response['body'].read())
        return response_body['content'][0]['text']
    
    except Exception as e:
        print(f"Error invoking model: {str(e)}")
        return None

# Usage
CUSTOM_MODEL_ARN = "arn:aws:bedrock:us-east-1:123456789012:custom-model/customer-support-model"

response = invoke_custom_model(
    CUSTOM_MODEL_ARN,
    "How do I track my order?"
)

print(f"Model Response:\n{response}")
```

**Option 2: Provisioned Throughput** (Required for some models)

Some models require provisioned throughput for custom model inference:

```python
def create_provisioned_throughput(custom_model_arn: str):
    """Create provisioned throughput for custom model"""
    
    bedrock = boto3.client('bedrock', region_name='us-east-1')
    
    response = bedrock.create_provisioned_model_throughput(
        modelUnits=1,  # Minimum is 1
        provisionedModelName=f"provisioned-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        modelId=custom_model_arn
    )
    
    throughput_arn = response['provisionedModelArn']
    print(f"Provisioned Throughput ARN: {throughput_arn}")
    print("Note: Provisioning takes 10-20 minutes")
    
    return throughput_arn

# Wait for provisioning to complete
def wait_for_provisioning(throughput_arn):
    """Wait for provisioned throughput to be ready"""
    
    bedrock = boto3.client('bedrock', region_name='us-east-1')
    
    while True:
        response = bedrock.get_provisioned_model_throughput(
            provisionedModelId=throughput_arn
        )
        
        status = response['status']
        print(f"Status: {status}")
        
        if status == 'InService':
            print("✅ Provisioned throughput is ready!")
            return True
        elif status == 'Failed':
            print("❌ Provisioning failed")
            return False
        
        time.sleep(60)  # Check every minute
```

### Testing Your Model

**Interactive Test Script:**

```python
def interactive_test():
    """Interactive testing interface"""
    
    CUSTOM_MODEL_ARN = "arn:aws:bedrock:us-east-1:123456789012:custom-model/your-model"
    
    print("🤖 Custom Model Interactive Test")
    print("=" * 60)
    print("Type 'quit' to exit\n")
    
    while True:
        prompt = input("You: ").strip()
        
        if prompt.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not prompt:
            continue
        
        print("\nModel: ", end='', flush=True)
        response = invoke_custom_model(CUSTOM_MODEL_ARN, prompt)
        print(response)
        print()

# Run interactive test
interactive_test()
```

### Batch Testing

```python
def batch_test(test_prompts: List[str], custom_model_arn: str):
    """Test multiple prompts and save results"""
    
    results = []
    
    print(f"🧪 Running batch test with {len(test_prompts)} prompts\n")
    
    for idx, prompt in enumerate(test_prompts, 1):
        print(f"[{idx}/{len(test_prompts)}] Testing: {prompt[:50]}...")
        
        start_time = time.time()
        response = invoke_custom_model(custom_model_arn, prompt)
        latency = time.time() - start_time
        
        results.append({
            'prompt': prompt,
            'response': response,
            'latency_seconds': latency
        })
        
        time.sleep(1)  # Rate limiting
    
    # Save results
    with open('batch_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Batch test complete! Results saved to batch_test_results.json")
    
    # Print summary
    avg_latency = sum(r['latency_seconds'] for r in results) / len(results)
    print(f"Average Response Time: {avg_latency:.2f} seconds")
    
    return results

# Example test prompts
test_prompts = [
    "How do I track my order?",
    "What is your return policy?",
    "Do you ship internationally?",
    "My package arrived damaged",
    "Can I change my shipping address?",
    "What payment methods do you accept?",
    "How do I reset my password?",
    "Do you offer gift wrapping?"
]

results = batch_test(test_prompts, CUSTOM_MODEL_ARN)
```

---

## Troubleshooting

### Common Issues and Solutions

#### Issue 1: "Invalid Training Data Format"

**Error Message:**
```
ValidationException: Training data validation failed: Invalid JSON format at line 42
```

**Solution:**
```python
# Validate your JSONL file
def find_json_errors(filepath):
    """Find lines with invalid JSON"""
    
    errors = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f, 1):
            try:
                json.loads(line)
            except json.JSONDecodeError as e:
                errors.append((idx, str(e), line[:100]))
    
    if errors:
        print("❌ JSON Errors Found:")
        for line_num, error, content in errors:
            print(f"Line {line_num}: {error}")
            print(f"  Content: {content}...")
    else:
        print("✅ All lines are valid JSON")
    
    return errors

find_json_errors('training_data.jsonl')
```

#### Issue 2: "Insufficient Training Data"

**Error Message:**
```
ValidationException: Training dataset must contain at least 32 examples
```

**Solution:**
- Collect more examples (minimum 32, recommended 100+)
- Use data augmentation:

```python
def augment_data(original_examples):
    """Create variations of existing examples"""
    
    augmented = []
    
    for example in original_examples:
        # Original
        augmented.append(example)
        
        # Variation 1: Rephrase question
        variations = {
            "How do I": "How can I",
            "What is": "What's",
            "Can you": "Could you"
        }
        
        prompt = example['prompt']
        for old, new in variations.items():
            if old in prompt:
                augmented.append({
                    'prompt': prompt.replace(old, new),
                    'completion': example['completion']
                })
    
    return augmented
```

#### Issue 3: Model Not Improving Performance

**Symptoms:**
- Validation loss higher than training loss
- Model responses not better than base model
- Poor generalization to new examples

**Solutions:**

1. **Check for Overfitting:**
```python
# If val_loss >> train_loss, reduce epochs
hyperParameters={
    'epochCount': '2',  # Reduce from 5
    'batchSize': '1',
    'learningRate': '0.00001'
}
```

2. **Improve Data Quality:**
```python
def check_data_quality(filepath):
    """Check for common data quality issues"""
    
    issues = []
    
    with jsonlines.open(filepath) as reader:
        prompts = []
        completions = []
        
        for line in reader:
            prompts.append(line['prompt'])
            completions.append(line['completion'])
        
        # Check for duplicates
        if len(prompts) != len(set(prompts)):
            issues.append("⚠️ Duplicate prompts found")
        
        # Check for very short completions
        short = sum(1 for c in completions if len(c) < 20)
        if short > len(completions) * 0.1:
            issues.append(f"⚠️ {short} completions are very short (<20 chars)")
        
        # Check for very long completions
        long = sum(1 for c in completions if len(c) > 2000)
        if long > 0:
            issues.append(f"⚠️ {long} completions exceed 2000 characters")
    
    if issues:
        print("Data Quality Issues:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("✅ Data quality looks good")
    
    return len(issues) == 0

check_data_quality('training_data.jsonl')
```

3. **Balance Your Dataset:**
```python
def balance_dataset(examples, max_per_category=50):
    """Balance examples across different categories"""
    
    from collections import defaultdict
    import random
    
    # Categorize by topic (example heuristic)
    categories = defaultdict(list)
    
    for ex in examples:
        # Simple categorization by first word
        category = ex['prompt'].split()[0].lower()
        categories[category].append(ex)
    
    # Sample evenly from each category
    balanced = []
    for category, items in categories.items():
        sample_size = min(len(items), max_per_category)
        balanced.extend(random.sample(items, sample_size))
    
    random.shuffle(balanced)
    return balanced
```

#### Issue 4: High Costs

**Problem:** Fine-tuning and inference costs are high

**Solutions:**

1. **Optimize Training Data Size:**
```python
# More data isn't always better
# 200-500 high-quality examples often sufficient
def optimal_dataset_size(total_examples):
    """Recommend optimal size"""
    if total_examples < 100:
        return total_examples, "Need more data"
    elif total_examples <= 500:
        return total_examples, "Good size"
    else:
        return 500, "Consider sampling 500 best examples"
```

2. **Use Provisioned Throughput Efficiently:**
```python
# Purchase provisioned throughput only when needed
# Use on-demand for testing and development
def cost_comparison():
    """Compare on-demand vs provisioned costs"""
    
    on_demand_per_1k_tokens = 0.003  # Example pricing
    provisioned_per_month = 500      # Example pricing
    
    monthly_tokens = 10_000_000  # 10M tokens/month
    
    on_demand_cost = (monthly_tokens / 1000) * on_demand_per_1k_tokens
    provisioned_cost = provisioned_per_month
    
    print(f"On-Demand Cost: ${on_demand_cost:.2f}/month")
    print(f"Provisioned Cost: ${provisioned_cost:.2f}/month")
    
    if provisioned_cost < on_demand_cost:
        print("✅ Provisioned is more cost-effective")
    else:
        print("✅ On-demand is more cost-effective")
```

#### Issue 5: Job Fails During Training

**Error:** Job status shows "Failed"

**Debugging Steps:**

```python
def diagnose_failed_job(job_arn):
    """Get detailed failure information"""
    
    bedrock = boto3.client('bedrock', region_name='us-east-1')
    
    response = bedrock.get_model_customization_job(
        jobIdentifier=job_arn
    )
    
    print("🔍 Job Failure Analysis")
    print("=" * 60)
    
    if 'failureMessage' in response:
        print(f"\nFailure Message:\n{response['failureMessage']}")
    
    # Check output location for logs
    if 'outputDataConfig' in response:
        output_uri = response['outputDataConfig']['s3Uri']
        print(f"\nCheck logs at: {output_uri}")
        print("\nTo download logs:")
        print(f"aws s3 cp {output_uri} ./logs/ --recursive")
    
    # Common failure reasons
    print("\n❓ Common Causes:")
    print("1. Insufficient IAM permissions for S3 access")
    print("2. Invalid training data format")
    print("3. Training data in wrong S3 location")
    print("4. Exceeded service quotas")
    
    return response

# Usage
diagnose_failed_job("arn:aws:bedrock:us-east-1:123456789012:model-customization-job/xyz")
```

---

## Cleanup

### Deleting Custom Models

```bash
# Bash/Linux/macOS
# List custom models
aws bedrock list-custom-models

# Delete a custom model
aws bedrock delete-custom-model \
    --model-identifier "arn:aws:bedrock:us-east-1:123456789012:custom-model/your-model"

# Delete provisioned throughput (if created)
aws bedrock delete-provisioned-model-throughput \
    --provisioned-model-id "arn:aws:bedrock:us-east-1:123456789012:provisioned-model/your-throughput"
```

```powershell
# PowerShell/Windows
# List custom models
aws bedrock list-custom-models

# Delete a custom model
aws bedrock delete-custom-model `
    --model-identifier "arn:aws:bedrock:us-east-1:123456789012:custom-model/your-model"

# Delete provisioned throughput (if created)
aws bedrock delete-provisioned-model-throughput `
    --provisioned-model-id "arn:aws:bedrock:us-east-1:123456789012:provisioned-model/your-throughput"
```

### Cleanup Python Script

```python
import boto3

def cleanup_resources():
    """Clean up all fine-tuning resources"""
    
    bedrock = boto3.client('bedrock', region_name='us-east-1')
    s3 = boto3.client('s3')
    
    print("🧹 Starting cleanup process...")
    
    # 1. List and delete custom models
    print("\n1️⃣ Cleaning up custom models...")
    models = bedrock.list_custom_models()
    
    for model in models.get('modelSummaries', []):
        model_arn = model['modelArn']
        model_name = model['modelName']
        
        confirm = input(f"Delete model '{model_name}'? (yes/no): ")
        if confirm.lower() == 'yes':
            try:
                bedrock.delete_custom_model(modelIdentifier=model_arn)
                print(f"✅ Deleted: {model_name}")
            except Exception as e:
                print(f"❌ Error deleting {model_name}: {str(e)}")
    
    # 2. List and delete provisioned throughput
    print("\n2️⃣ Cleaning up provisioned throughput...")
    throughputs = bedrock.list_provisioned_model_throughputs()
    
    for pt in throughputs.get('provisionedModelSummaries', []):
        pt_arn = pt['provisionedModelArn']
        pt_name = pt['provisionedModelName']
        
        confirm = input(f"Delete provisioned throughput '{pt_name}'? (yes/no): ")
        if confirm.lower() == 'yes':
            try:
                bedrock.delete_provisioned_model_throughput(
                    provisionedModelId=pt_arn
                )
                print(f"✅ Deleted: {pt_name}")
            except Exception as e:
                print(f"❌ Error deleting {pt_name}: {str(e)}")
    
    # 3. Clean up S3 bucket
    print("\n3️⃣ Cleaning up S3 data...")
    bucket_name = input("Enter S3 bucket name to clean (or 'skip'): ")
    
    if bucket_name.lower() != 'skip':
        confirm = input(f"Delete ALL contents from '{bucket_name}'? (yes/no): ")
        if confirm.lower() == 'yes':
            try:
                # List and delete all objects
                objects = s3.list_objects_v2(Bucket=bucket_name)
                
                if 'Contents' in objects:
                    for obj in objects['Contents']:
                        s3.delete_object(Bucket=bucket_name, Key=obj['Key'])
                        print(f"  Deleted: {obj['Key']}")
                
                # Delete bucket
                s3.delete_bucket(Bucket=bucket_name)
                print(f"✅ Deleted bucket: {bucket_name}")
            except Exception as e:
                print(f"❌ Error deleting bucket: {str(e)}")
    
    # 4. Delete IAM role
    print("\n4️⃣ Cleaning up IAM role...")
    role_name = "BedrockFineTuningRole"
    
    confirm = input(f"Delete IAM role '{role_name}'? (yes/no): ")
    if confirm.lower() == 'yes':
        iam = boto3.client('iam')
        try:
            # Delete inline policies first
            policies = iam.list_role_policies(RoleName=role_name)
            for policy in policies.get('PolicyNames', []):
                iam.delete_role_policy(RoleName=role_name, PolicyName=policy)
            
            # Delete role
            iam.delete_role(RoleName=role_name)
            print(f"✅ Deleted role: {role_name}")
        except Exception as e:
            print(f"❌ Error deleting role: {str(e)}")
    
    print("\n✅ Cleanup complete!")

# Run cleanup
cleanup_resources()
```

---

## Cost Considerations

### Fine-Tuning Costs (February 2026 Estimates)

**Training Costs:**
| Model | Price per 1000 tokens | Typical Job Cost |
|-------|----------------------|------------------|
| Claude 3 Haiku | $0.012 | $20-$100 |
| Titan Text Express | $0.008 | $15-$75 |
| Cohere Command | $0.010 | $18-$90 |

**Storage Costs:**
- S3 Standard: $0.023 per GB/month
- Typical dataset: $0.01-$0.05/month

**Inference Costs:**
- On-Demand: ~2-3x base model cost
- Provisioned Throughput: Fixed monthly fee ($500-$2000+ depending on model units)

### Cost Optimization Tips

1. **Start Small:** Test with 100-200 examples before scaling
2. **Use Validation Data:** Catch issues early to avoid wasting training runs
3. **Monitor Training Loss:** Stop training early if loss plateaus
4. **Delete Unused Models:** Don't keep old custom models you're not using
5. **Provisioned vs On-Demand:** Calculate break-even point for your usage

```python
def calculate_breakeven_point():
    """Calculate when provisioned is cheaper than on-demand"""
    
    on_demand_per_1k = 0.003  # Example
    provisioned_monthly = 500  # Example
    
    # Calculate tokens needed for break-even
    breakeven_tokens = (provisioned_monthly / on_demand_per_1k) * 1000
    
    print(f"Break-even point: {breakeven_tokens:,.0f} tokens/month")
    print(f"That's approximately {breakeven_tokens/1000:,.0f}K tokens/month")
    
    # Daily average
    daily_tokens = breakeven_tokens / 30
    print(f"Daily average: {daily_tokens:,.0f} tokens/day")

calculate_breakeven_point()
```

---

## Next Steps

### 1. Advanced Fine-Tuning Techniques
- **Multi-task Fine-Tuning:** Train on multiple tasks simultaneously
- **Continual Fine-Tuning:** Update models with new data over time
- **Few-Shot Learning:** Combine fine-tuning with in-context examples

### 2. Production Deployment
- Set up monitoring with CloudWatch
- Implement A/B testing framework
- Create automated retraining pipelines
- Build feedback collection systems

### 3. Integration Patterns
- Connect with Lambda for serverless inference
- Build API Gateway endpoints
- Integrate with Step Functions for workflows
- Set up EventBridge for automated triggers

### 4. Further Learning Resources
- **AWS Documentation:** [Bedrock Fine-Tuning Guide](https://docs.aws.amazon.com/bedrock/)
- **AWS Workshops:** Search for "Bedrock Fine-Tuning Workshop"
- **AWS Blog:** Latest updates and best practices
- **Community Forums:** AWS re:Post for Q&A

---

## Summary

You've learned how to:
- ✅ Understand when fine-tuning is the right approach
- ✅ Prepare high-quality training datasets in JSONL format
- ✅ Create and monitor fine-tuning jobs in AWS Bedrock
- ✅ Evaluate model performance with metrics and testing
- ✅ Deploy and use custom models in production
- ✅ Troubleshoot common issues
- ✅ Optimize costs and manage resources

**Key Takeaways:**
1. Fine-tuning requires quality over quantity - 200 good examples beat 2000 mediocre ones
2. Always validate your data format before starting training
3. Monitor validation loss to prevent overfitting
4. Test thoroughly before production deployment
5. Consider costs when choosing provisioned vs on-demand

**Ready for hands-on practice?** Try fine-tuning a model for your specific use case following this guide!

---

*Tutorial created: February 2026*  
*For questions or issues, please refer to the troubleshooting section or AWS documentation.*
