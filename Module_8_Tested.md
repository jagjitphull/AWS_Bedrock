# AWS Bedrock Fine-Tuning Automation — Beginner's Complete Guide

> **Who is this for?** Absolute beginners to AI and Cloud. No prior AWS or machine learning experience required.  
> **Time Commitment:** ~8–10 hours across 2 weeks (broken into manageable sessions)  
> **Cost Estimate:** ~$15–$40 USD in AWS charges (we'll show you how to keep costs low)

---

## What You'll Learn (And Why It Matters)

Imagine you hire a brilliant new employee. They're smart, but they don't know *your* company's language, products, or customers yet. **Fine-tuning** is like giving that employee specialised on-the-job training so they become an expert in *your* specific domain.

In this guide, you'll learn how to:

1. **Prepare training data** — Create the "textbook" your AI model will learn from
2. **Fine-tune an AI model** — Teach Amazon Bedrock's foundation models your specific use case
3. **Automate the entire process** — Build pipelines that retrain your model automatically when new data arrives
4. **Evaluate the results** — Scientifically compare "before" and "after" to prove your fine-tuning worked

By the end, you'll have a fully automated AI training pipeline — the same kind of system enterprise teams build in production.

---

## Prerequisites

### Accounts & Access

| Requirement | What It Is | How to Get It |
|---|---|---|
| **AWS Account** | Your cloud workspace | [Create a free account](https://aws.amazon.com/free/) |
| **AWS CLI** | Command-line tool to talk to AWS | [Installation guide](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) |
| **Python 3.9+** | Programming language we'll use | [Download Python](https://www.python.org/downloads/) |
| **Bedrock Model Access** | Permission to use AI models | We'll walk through this below |

### Concepts You Should Know (We'll Explain as We Go)

Don't worry if these terms are new — think of this as a preview, not a test:

- **Foundation Model** — A pre-trained AI model (like Claude or Titan) that already knows a lot, but hasn't been customised for your needs. Think of it as a university graduate who is smart but hasn't started their first job yet.
- **Fine-Tuning** — Teaching that foundation model your specific domain. Like that graduate completing a 3-month internship at your company.
- **Pipeline** — A series of automated steps that run in order, like a factory assembly line. Data goes in one end, a trained model comes out the other.
- **Orchestration** — Coordinating multiple steps so they run in the right order, handle errors, and notify you when done. Think of it as the factory floor manager.

---

## Environment Setup

### Step 1 — Install Python Dependencies

Open your terminal (Command Prompt on Windows, Terminal on Mac/Linux) and run:

```bash
# Create a project folder
mkdir bedrock-finetuning-lab
cd bedrock-finetuning-lab

# Create a virtual environment (keeps your project isolated — like a clean workspace)
python -m venv venv

# Activate it
# On Mac/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install the libraries we need
pip install boto3 rich
```

**What did we just install?**
- `boto3` — The official Python toolkit for talking to AWS services. Think of it as your translator between Python and the cloud.
- `rich` — Makes our terminal output colourful and easy to read (because learning should be fun).

### Step 2 — Configure AWS Credentials

```bash
aws configure
```

You'll be prompted to enter four things:

```
AWS Access Key ID:     [Your access key]
AWS Secret Access Key: [Your secret key]
Default region name:   ap-south-1
Default output format: json
```

**Where do I find my keys?** Go to the AWS Console → Click your username (top right) → Security credentials → Create access key.

> **Security Note:** Treat your secret key like a password. Never share it, never commit it to GitHub, never paste it in a chat.

### Step 3 — Enable Bedrock Model Access

Before you can fine-tune models, AWS needs you to explicitly request access:

1. Open the [Amazon Bedrock Console](https://console.aws.amazon.com/bedrock/)
2. In the left sidebar, click **Model access**
3. Click **Manage model access**
4. Enable access for:
   - **Amazon Titan Text Express** (we'll use this for fine-tuning)
   - **Anthropic Claude 3.5 Sonnet** (we'll use this for evaluation)
5. Click **Save changes**

> **Why two models?** We fine-tune the Titan model (because Bedrock supports custom training for it), and we use Claude to *evaluate* whether the fine-tuned model actually improved. It's like having a teacher (Claude) grade the student's (Titan's) homework.

Model access approval usually takes a few minutes but can take up to 24 hours.

### Step 4 — Create an S3 Bucket for Training Data

S3 (Simple Storage Service) is like a filing cabinet in the cloud where we'll store our training data and model outputs.

```bash
# Replace 'your-unique-name' with something unique to you
aws s3 mb s3://bedrock-finetuning-your-unique-name --region ap-south-1
```

> **Naming tip:** S3 bucket names must be globally unique across ALL of AWS. Try including your name or a random number, like `bedrock-finetuning-jane-2025`.

### Step 5 — Verify Everything Works

Create a file called `verify_setup.py` and paste this in:

```python
import boto3
import json

def check_setup():
    print("=" * 50)
    print("  AWS Bedrock Fine-Tuning — Setup Verification")
    print("=" * 50)

    # Check 1: Can we connect to AWS?
    try:
        sts = boto3.client("sts")
        identity = sts.get_caller_identity()
        print(f"\n✅ AWS Connection     : Working")
        print(f"   Account ID        : {identity['Account']}")
    except Exception as e:
        print(f"\n❌ AWS Connection     : Failed — {e}")
        return

    # Check 2: Can we reach Bedrock?
    try:
        bedrock = boto3.client("bedrock", region_name="ap-south-1")
        models = bedrock.list_foundation_models()
        model_count = len(models.get("modelSummaries", []))
        print(f"✅ Bedrock Access     : Working ({model_count} models available)")
    except Exception as e:
        print(f"❌ Bedrock Access     : Failed — {e}")
        return

    # Check 3: Can we access S3?
    try:
        s3 = boto3.client("s3")
        buckets = s3.list_buckets()
        bucket_names = [b["Name"] for b in buckets["Buckets"]]
        ft_buckets = [b for b in bucket_names if "finetuning" in b.lower()]
        if ft_buckets:
            print(f"✅ S3 Bucket          : Found — {ft_buckets[0]}")
        else:
            print(f"⚠️  S3 Bucket          : No fine-tuning bucket found (create one in Step 4)")
    except Exception as e:
        print(f"❌ S3 Access          : Failed — {e}")

    print("\n" + "=" * 50)
    print("  Setup verification complete!")
    print("=" * 50)

if __name__ == "__main__":
    check_setup()
```

Run it:

```bash
python verify_setup.py
```

You should see green checkmarks (✅) for all three checks. If you see any red (❌), revisit the corresponding step above.

---

## Part 1 — Understanding Automating Fine-Tuning

Before we get hands-on, let's understand the building blocks. Each section below covers a key concept, followed by its hands-on lab.

### 1.1 — Step Functions for Orchestrating Training

**The Real-World Analogy:**  
Imagine you're baking a cake. You can't frost it before it's baked, and you can't bake it before mixing the ingredients. AWS Step Functions is your recipe card — it defines each step, the order they run in, and what to do if something goes wrong (like the oven breaking).

**What Step Functions Does for Fine-Tuning:**

Step Functions is an AWS service that lets you build **state machines** — visual workflows where each "state" is a step in your process. For fine-tuning, a typical workflow looks like this:

```
[Validate Data] → [Start Fine-Tuning Job] → [Wait for Completion] → [Run Evaluation] → [Register Model] → [Notify Team]
```

Each box is a step. Step Functions handles:
- **Sequencing** — Steps run in the correct order, automatically
- **Error handling** — If a step fails, it can retry or take an alternative path
- **Waiting** — Fine-tuning can take hours. Step Functions patiently waits without you paying for idle compute
- **Visibility** — A visual dashboard shows you exactly which step is running, which passed, and which failed

**Key Terms:**
- **State Machine** — The complete workflow definition (your entire recipe)
- **State** — A single step in the workflow (one instruction in the recipe)
- **Task State** — A state that does work (like calling Bedrock to start fine-tuning)
- **Wait State** — A state that pauses (waiting for the fine-tuning job to finish)
- **Choice State** — A decision point (if evaluation score > 0.8, deploy; otherwise, alert the team)

### 1.2 — Using EventBridge for Automation

**The Real-World Analogy:**  
EventBridge is like setting up an automatic rule in your email: "When I receive an email from my boss with the word 'urgent', forward it to my phone." EventBridge watches for **events** (things that happen in AWS) and automatically **triggers actions** in response.

**What EventBridge Does for Fine-Tuning:**

Instead of manually clicking "start fine-tuning" every time you have new data, EventBridge can:

- **Detect new training data** — "When a new file is uploaded to my S3 training bucket…"
- **Run on a schedule** — "Every Sunday at 2 AM, check for new data and retrain…"
- **React to job completion** — "When a fine-tuning job finishes, start the evaluation pipeline…"

**Common EventBridge Patterns for Fine-Tuning:**

| Trigger | Action | Use Case |
|---|---|---|
| New file uploaded to S3 | Start Step Functions workflow | Automatic retraining on new data |
| Cron schedule (weekly) | Start Step Functions workflow | Regular retraining cadence |
| Bedrock job status change | Send SNS notification | Alert team when training finishes |
| Evaluation score below threshold | Trigger alert / rollback | Catch quality regressions |

**Key Terms:**
- **Event** — A record of something that happened ("file uploaded", "job completed")
- **Rule** — The condition you define ("when event X happens, do Y")
- **Target** — What gets triggered (Step Functions, Lambda, SNS, etc.)
- **Event Bus** — The central highway where events flow through

### 1.3 — Automated Evaluation Pipeline

**The Real-World Analogy:**  
After training a new employee, you don't just *hope* they learned — you give them a test. An automated evaluation pipeline is that test for your fine-tuned model. It runs automatically after every training job and tells you: "Did the model actually get better?"

**How Evaluation Works:**

```
[Fine-Tuning Complete] → [Load Test Dataset] → [Run Baseline Model] → [Run Fine-Tuned Model] → [Compare Scores] → [Generate Report]
```

A good evaluation pipeline measures multiple dimensions:

- **Accuracy** — Does the model give correct answers?
- **Relevance** — Are the answers on-topic and useful?
- **Format Compliance** — Does the model follow your desired output format?
- **Safety** — Does the model still refuse harmful requests?

**Why Automate Evaluation?**  
Manual evaluation is slow, inconsistent, and doesn't scale. Automated evaluation runs the *exact same tests* every time, giving you comparable results across model versions. This is critical when you need to answer: "Is version 3 better than version 2?"

### 1.4 — Model Versioning & Model Registry Strategy

**The Real-World Analogy:**  
Think of this like version control for your AI models — just like how Google Docs saves every version of your document so you can go back to last Tuesday's version if today's edits were bad.

**Why Model Versioning Matters:**

Without versioning, you have questions like:
- "Which model is running in production right now?"
- "The model was better last week — can we go back?"
- "Which training data produced this model?"

**A Practical Versioning Strategy:**

```
models/
├── banking-assistant-v1.0/     ← Baseline (original foundation model)
│   ├── model-id.txt
│   ├── training-config.json
│   └── evaluation-report.json
├── banking-assistant-v1.1/     ← First fine-tune
│   ├── model-id.txt
│   ├── training-config.json
│   ├── training-data-manifest.json
│   └── evaluation-report.json
└── banking-assistant-v2.0/     ← Major improvement
    ├── model-id.txt
    ├── training-config.json
    ├── training-data-manifest.json
    └── evaluation-report.json
```

**What to Track for Each Version:**

| Artefact | Why | Example |
|---|---|---|
| Model ID | So you can use it in Bedrock | `arn:aws:bedrock:ap-south-1:123456:custom-model/...` |
| Training Config | Reproducibility — retrain the same way | Epochs, learning rate, batch size |
| Training Data Reference | Know what the model learned from | S3 path + row count + date range |
| Evaluation Scores | Compare versions objectively | Accuracy: 87%, Relevance: 92% |
| Deployment Date | Audit trail | 2025-07-15T10:30:00Z |
| Rollback Pointer | Quick recovery if something goes wrong | "Previous version: v1.1" |

**Registry Pattern in Practice:**

We'll use a simple JSON-based registry stored in S3. Enterprise teams might use SageMaker Model Registry or a database, but the principle is identical: every model version gets a record with metadata.

---

## Part 2 — Hands-On Labs

Now let's build it. Each lab builds on the previous one, so complete them in order.

> **Pacing Tip:** Each lab is designed for a single focused session. Take breaks between labs — the concepts need time to settle.

---

### Lab 1 — Prepare a Fine-Tuning Dataset (Instruction-Tuning Style)

**Time:** ~1.5 hours  
**Goal:** Create a properly formatted training dataset for Amazon Bedrock fine-tuning  
**Scenario:** You're building a banking customer service assistant

#### What Is Instruction Tuning?

Standard fine-tuning says: "Here's a lot of text — learn from it."  
Instruction tuning says: "Here's a question → here's the ideal answer. Learn to respond like this."

The format looks like this:

```json
{
    "prompt": "What is the process to dispute a credit card transaction?",
    "completion": "To dispute a credit card transaction, follow these steps: 1) Contact our support team within 60 days of the transaction date. 2) Provide the transaction date, amount, and merchant name. 3) We will issue a temporary credit while investigating. 4) Investigation typically completes within 10 business days. 5) You will receive written notification of the outcome."
}
```

Each line is one example. You're teaching the model: "When someone asks X, respond with Y."

#### Step-by-Step

**1. Create your dataset file:**

Create a file called `create_training_data.py`:

```python
import json
import os

# Our training examples — each one teaches the model how to handle
# a specific type of banking customer query.
training_examples = [
    {
        "prompt": "What is the process to dispute a credit card transaction?",
        "completion": "To dispute a credit card transaction: First, contact our support team within 60 days of the transaction date. Provide the transaction date, amount, and merchant name. We will issue a temporary credit to your account while we investigate. The investigation typically takes 10 business days, and you will receive written notification of the outcome via email or mail."
    },
    {
        "prompt": "How do I set up automatic bill payments?",
        "completion": "To set up automatic bill payments: Log into your online banking portal and navigate to the Bill Pay section. Click Add New Payee and enter the company name, your account number with them, and the payment amount. Choose your payment frequency (weekly, bi-weekly, or monthly) and the start date. Review the details and confirm. You can modify or cancel automatic payments at any time from the same section."
    },
    {
        "prompt": "What should I do if I suspect fraudulent activity on my account?",
        "completion": "If you suspect fraud on your account, take these immediate steps: Call our 24/7 fraud hotline to temporarily freeze your account. This prevents any further unauthorized transactions. Review your recent transaction history and note any unfamiliar charges. We will cancel your current card and issue a replacement within 3-5 business days. File a fraud report, which we will investigate within 48 hours. You are not liable for unauthorized transactions reported within 2 business days."
    },
    {
        "prompt": "How do I apply for a personal loan?",
        "completion": "To apply for a personal loan: You can start your application online, by phone, or at any branch. You will need to provide proof of identity, proof of income (recent pay stubs or tax returns), and your employment details. Loan amounts range from 1,000 to 50,000 with terms from 12 to 60 months. We typically provide a decision within 1 business day. If approved, funds are deposited into your account within 2-3 business days."
    },
    {
        "prompt": "What are the fees for international wire transfers?",
        "completion": "International wire transfer fees depend on the transfer method and destination. Outgoing international wires have a flat fee of 45 for transfers sent in USD and 35 for transfers in foreign currency. Incoming international wires have a fee of 15. Processing time is typically 1-3 business days. Expedited same-day transfers are available for an additional 25. Exchange rates are set at the time of transfer and include a small conversion margin."
    },
    {
        "prompt": "How can I increase my credit card limit?",
        "completion": "To request a credit limit increase: Log into your account online and navigate to Credit Card Services, then select Request Credit Limit Increase. You can also call the number on the back of your card. We will review your account history, payment patterns, and current income. Automatic increases are periodically applied to accounts in good standing. A formal request may involve a soft credit inquiry that does not impact your credit score. Decisions are typically provided immediately or within 2 business days."
    },
    {
        "prompt": "What happens if I miss a mortgage payment?",
        "completion": "If you miss a mortgage payment, here is what to expect: A late fee is assessed if payment is not received within 15 days of the due date. After 30 days, the missed payment may be reported to credit bureaus, which can affect your credit score. Contact us immediately if you are having difficulty making payments because we offer hardship programs including temporary payment reduction, forbearance, and loan modification options. The sooner you reach out, the more options we can offer."
    },
    {
        "prompt": "How do I order new checks for my checking account?",
        "completion": "To order new checks: Log into your online banking and go to Account Services, then select Order Checks. You can choose from several check designs including standard, secure, and premium options. Standard checks are complimentary for premium account holders. For other accounts, pricing starts at 20 per box. Checks typically arrive within 7-10 business days. You can also order by calling customer service or visiting any branch."
    },
    {
        "prompt": "What is the difference between a savings account and a money market account?",
        "completion": "The key differences are: A savings account offers a fixed interest rate, has no minimum balance requirement for our basic tier, and allows up to 6 withdrawals per month. A money market account typically offers a higher interest rate that scales with your balance, requires a minimum balance of 2,500, and includes check-writing privileges and a debit card. Both accounts are FDIC insured up to 250,000. A money market account is better for larger balances where you want higher returns with some transaction flexibility."
    },
    {
        "prompt": "How do I set up two-factor authentication on my account?",
        "completion": "To enable two-factor authentication: Go to Settings in your online banking portal and select Security Preferences. Under Two-Factor Authentication, choose your preferred method: SMS text codes sent to your phone, authentication app (we support Google Authenticator and Authy), or a physical security key. Follow the on-screen prompts to verify your chosen method. Once enabled, you will need to provide a verification code each time you log in from an unrecognized device. We strongly recommend using an authentication app for the highest security."
    }
]

def create_dataset():
    """Create the JSONL training file for Bedrock fine-tuning."""

    output_file = "training_data.jsonl"

    print("=" * 55)
    print("  Banking Assistant — Training Data Generator")
    print("=" * 55)

    # Write each example as a separate JSON line (JSONL format)
    with open(output_file, "w") as f:
        for i, example in enumerate(training_examples):
            f.write(json.dumps(example) + "\n")
            print(f"  ✅ Example {i + 1:>2}: {example['prompt'][:50]}...")

    print(f"\n  📁 Created: {output_file}")
    print(f"  📊 Total examples: {len(training_examples)}")
    print(f"  📏 File size: {os.path.getsize(output_file):,} bytes")
    print("\n" + "=" * 55)

    # Validate the file
    print("\n  Running validation checks...")
    with open(output_file, "r") as f:
        lines = f.readlines()
        for line_num, line in enumerate(lines, 1):
            data = json.loads(line.strip())
            assert "prompt" in data, f"Line {line_num}: Missing 'prompt' field"
            assert "completion" in data, f"Line {line_num}: Missing 'completion' field"
            assert len(data["prompt"]) > 0, f"Line {line_num}: Empty prompt"
            assert len(data["completion"]) > 0, f"Line {line_num}: Empty completion"

    print("  ✅ All validation checks passed!")
    print("  ✅ File format: JSONL (one JSON object per line)")
    print("  ✅ All examples have 'prompt' and 'completion' fields")

    return output_file

if __name__ == "__main__":
    create_dataset()
```

Run it:

```bash
python create_training_data.py
```

**2. Upload the dataset to S3:**

```bash
# Replace with your actual bucket name from Setup Step 4
aws s3 cp training_data.jsonl s3://bedrock-finetuning-your-unique-name/training-data/training_data.jsonl
```

**3. Verify the upload:**

```bash
aws s3 ls s3://bedrock-finetuning-your-unique-name/training-data/
```

You should see your `training_data.jsonl` file listed.

#### What You Just Accomplished

You created an **instruction-tuning dataset** in JSONL format — the standard format Bedrock expects. Each line contains a prompt-completion pair that teaches the model a specific banking interaction pattern.

> **Real-World Note:** Production datasets typically have 100–10,000+ examples. Our 10 examples are enough to demonstrate the process. More examples generally produce better results, but quality matters more than quantity. Ten excellent examples beat a thousand sloppy ones.

---

### Lab 2 — Trigger a Fine-Tuning Job (Bedrock Custom Model)

**Time:** ~1.5 hours (plus waiting time for the job to complete)  
**Goal:** Submit a fine-tuning job to Amazon Bedrock and monitor its progress  
**Prerequisite:** Lab 1 completed (training data uploaded to S3)

#### Important: IAM Permissions

Bedrock needs permission to read your training data from S3. We need to create an IAM role for this.

**1. Create the trust policy file** — `bedrock_trust_policy.json`:

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

**2. Create the S3 access policy file** — `bedrock_s3_policy.json`:

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
                "arn:aws:s3:::bedrock-finetuning-your-unique-name",
                "arn:aws:s3:::bedrock-finetuning-your-unique-name/*"
            ]
        }
    ]
}
```

> **Remember:** Replace `bedrock-finetuning-your-unique-name` with your actual bucket name.

**3. Create the IAM role:**

```bash
# Create the role
aws iam create-role \
    --role-name BedrockFineTuningRole \
    --assume-role-policy-document file://bedrock_trust_policy.json

# Attach the S3 access policy
aws iam put-role-policy \
    --role-name BedrockFineTuningRole \
    --policy-name BedrockS3Access \
    --policy-document file://bedrock_s3_policy.json
```

**4. Create the fine-tuning script** — `start_finetuning.py`:

```python
import boto3
import json
import time
from datetime import datetime

# ─── CONFIGURATION ──────────────────────────────────────────────
# Update these values to match your setup
REGION = "ap-south-1"
BUCKET_NAME = "bedrock-finetuning-your-unique-name"  # ← Change this!
ROLE_NAME = "BedrockFineTuningRole"
BASE_MODEL_ID = "amazon.titan-text-express-v1"
# ─────────────────────────────────────────────────────────────────

def get_role_arn(role_name):
    """Look up the full ARN for our IAM role."""
    iam = boto3.client("iam")
    response = iam.get_role(RoleName=role_name)
    return response["Role"]["Arn"]

def start_fine_tuning_job():
    """Submit a fine-tuning job to Amazon Bedrock."""

    print("=" * 55)
    print("  Bedrock Fine-Tuning — Job Launcher")
    print("=" * 55)

    try:
        bedrock = boto3.client("bedrock", region_name=REGION)
        role_arn = get_role_arn(ROLE_NAME)

        # Generate a unique job name using timestamp
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        job_name = f"banking-assistant-ft-{timestamp}"
        custom_model_name = f"banking-assistant-{timestamp}"

        print(f"\n  📋 Job Name         : {job_name}")
        print(f"  🧠 Base Model       : {BASE_MODEL_ID}")
        print(f"  📁 Training Data    : s3://{BUCKET_NAME}/training-data/training_data.jsonl")
        print(f"  🔑 IAM Role         : {role_arn}")
        print(f"  🌏 Region           : {REGION}")

        # Define hyperparameters
        # These control HOW the model learns — don't worry about tuning
        # these for now; the defaults work well for learning purposes.
        hyper_parameters = {
            "epochCount": "2",          # How many times the model reads through all examples
            "batchSize": "1",           # How many examples to process at once
            "learningRate": "0.00001"   # How fast the model adjusts (smaller = more careful)
        }

        print(f"\n  ⚙️  Hyperparameters:")
        for key, value in hyper_parameters.items():
            print(f"     {key}: {value}")

        print(f"\n  🚀 Submitting fine-tuning job...")
        response = bedrock.create_model_customization_job(
            jobName=job_name,
            customModelName=custom_model_name,
            roleArn=role_arn,
            baseModelIdentifier=BASE_MODEL_ID,
            trainingDataConfig={
                "s3Uri": f"s3://{BUCKET_NAME}/training-data/training_data.jsonl"
            },
            outputDataConfig={
                "s3Uri": f"s3://{BUCKET_NAME}/output/"
            },
            hyperParameters=hyper_parameters
        )

        job_arn = response["jobArn"]
        print(f"\n  ✅ Job submitted successfully!")
        print(f"  📌 Job ARN: {job_arn}")

        # Save job details for later use
        job_details = {
            "job_name": job_name,
            "job_arn": job_arn,
            "custom_model_name": custom_model_name,
            "base_model_id": BASE_MODEL_ID,
            "submitted_at": datetime.now().isoformat(),
            "bucket": BUCKET_NAME
        }

        with open("job_details.json", "w") as f:
            json.dump(job_details, f, indent=2)

        print(f"  💾 Job details saved to: job_details.json")

        # Monitor the job
        print(f"\n  ⏳ Monitoring job status (press Ctrl+C to stop monitoring)...")
        print(f"     Fine-tuning typically takes 30 minutes to 2+ hours.\n")

        while True:
            status_response = bedrock.get_model_customization_job(jobIdentifier=job_arn)
            status = status_response["status"]
            elapsed = datetime.now().strftime("%H:%M:%S")

            if status == "Completed":
                print(f"  [{elapsed}] ✅ COMPLETED — Your fine-tuned model is ready!")
                print(f"\n  🎉 Custom Model Name: {custom_model_name}")

                # Update job details
                job_details["status"] = "Completed"
                job_details["completed_at"] = datetime.now().isoformat()
                with open("job_details.json", "w") as f:
                    json.dump(job_details, f, indent=2)
                break

            elif status == "Failed":
                failure_message = status_response.get("failureMessage", "Unknown error")
                print(f"  [{elapsed}] ❌ FAILED — {failure_message}")
                break

            elif status == "Stopped":
                print(f"  [{elapsed}] ⏹️  STOPPED — Job was manually cancelled")
                break

            else:
                print(f"  [{elapsed}] ⏳ Status: {status}...")
                time.sleep(60)  # Check every 60 seconds

    except Exception as e:
        print(f"\n  ❌ Error: {e}")
        print(f"\n  💡 Troubleshooting tips:")
        print(f"     1. Verify your bucket name is correct")
        print(f"     2. Ensure the IAM role has the correct permissions")
        print(f"     3. Check that model access is enabled in Bedrock console")
        print(f"     4. Verify the training data file exists in S3")

    print("\n" + "=" * 55)

if __name__ == "__main__":
    start_fine_tuning_job()
```

Run it:

```bash
python start_finetuning.py
```

> **While you wait:** Fine-tuning takes time (30 minutes to 2+ hours). This is normal — the model is literally learning from your examples. Use this time to read ahead to Lab 3 and Lab 4, or take a well-deserved break.

#### What You Just Accomplished

You submitted a **custom model training job** to Amazon Bedrock. Behind the scenes, AWS is spinning up specialised hardware, loading the Titan foundation model, and adjusting its neural network weights based on your banking examples. When it finishes, you'll have a model that's better at banking-related conversations than the generic version.

---

### Lab 3 — Automate Retraining Using EventBridge + Step Functions

**Time:** ~2.5 hours  
**Goal:** Build an automated pipeline that retrains your model when new data arrives  
**Prerequisite:** Labs 1 and 2 completed

This is where it gets powerful. Instead of manually running scripts, we'll build a system that **automatically detects new training data and kicks off the entire fine-tuning pipeline**.

#### Architecture Overview

Here's what we're building:

```
New file uploaded      EventBridge          Step Functions         Bedrock
to S3 bucket      →   detects the      →   orchestrates the  →   fine-tunes
                       upload event         pipeline steps         the model
```

#### Step 1 — Create the Step Functions State Machine

The state machine defines our pipeline. Create `step_function_definition.json`:

```json
{
    "Comment": "Bedrock Fine-Tuning Automation Pipeline",
    "StartAt": "ValidateTrainingData",
    "States": {
        "ValidateTrainingData": {
            "Type": "Task",
            "Resource": "arn:aws:states:::lambda:invoke",
            "Parameters": {
                "FunctionName": "validate-training-data",
                "Payload": {
                    "bucket.$": "$.bucket",
                    "key.$": "$.key"
                }
            },
            "ResultPath": "$.validation",
            "Next": "CheckValidation",
            "Catch": [
                {
                    "ErrorEquals": ["States.ALL"],
                    "Next": "NotifyFailure",
                    "ResultPath": "$.error"
                }
            ]
        },
        "CheckValidation": {
            "Type": "Choice",
            "Choices": [
                {
                    "Variable": "$.validation.Payload.isValid",
                    "BooleanEquals": true,
                    "Next": "StartFineTuningJob"
                }
            ],
            "Default": "NotifyValidationFailed"
        },
        "StartFineTuningJob": {
            "Type": "Task",
            "Resource": "arn:aws:states:::lambda:invoke",
            "Parameters": {
                "FunctionName": "start-bedrock-finetuning",
                "Payload": {
                    "bucket.$": "$.bucket",
                    "key.$": "$.key",
                    "validation.$": "$.validation"
                }
            },
            "ResultPath": "$.finetuning",
            "Next": "WaitForTraining"
        },
        "WaitForTraining": {
            "Type": "Wait",
            "Seconds": 300,
            "Next": "CheckTrainingStatus"
        },
        "CheckTrainingStatus": {
            "Type": "Task",
            "Resource": "arn:aws:states:::lambda:invoke",
            "Parameters": {
                "FunctionName": "check-finetuning-status",
                "Payload": {
                    "jobArn.$": "$.finetuning.Payload.jobArn"
                }
            },
            "ResultPath": "$.status",
            "Next": "IsTrainingComplete"
        },
        "IsTrainingComplete": {
            "Type": "Choice",
            "Choices": [
                {
                    "Variable": "$.status.Payload.status",
                    "StringEquals": "Completed",
                    "Next": "RunEvaluation"
                },
                {
                    "Variable": "$.status.Payload.status",
                    "StringEquals": "Failed",
                    "Next": "NotifyFailure"
                }
            ],
            "Default": "WaitForTraining"
        },
        "RunEvaluation": {
            "Type": "Task",
            "Resource": "arn:aws:states:::lambda:invoke",
            "Parameters": {
                "FunctionName": "evaluate-finetuned-model",
                "Payload": {
                    "customModelName.$": "$.finetuning.Payload.customModelName",
                    "baseModelId": "amazon.titan-text-express-v1"
                }
            },
            "ResultPath": "$.evaluation",
            "Next": "RegisterModel"
        },
        "RegisterModel": {
            "Type": "Task",
            "Resource": "arn:aws:states:::lambda:invoke",
            "Parameters": {
                "FunctionName": "register-model-version",
                "Payload": {
                    "customModelName.$": "$.finetuning.Payload.customModelName",
                    "evaluation.$": "$.evaluation.Payload"
                }
            },
            "ResultPath": "$.registry",
            "Next": "NotifySuccess"
        },
        "NotifySuccess": {
            "Type": "Task",
            "Resource": "arn:aws:states:::sns:publish",
            "Parameters": {
                "TopicArn": "arn:aws:sns:ap-south-1:ACCOUNT_ID:finetuning-notifications",
                "Subject": "Fine-Tuning Complete",
                "Message.$": "States.Format('Model {} training complete. Evaluation score: {}', $.finetuning.Payload.customModelName, $.evaluation.Payload.overallScore)"
            },
            "End": true
        },
        "NotifyValidationFailed": {
            "Type": "Task",
            "Resource": "arn:aws:states:::sns:publish",
            "Parameters": {
                "TopicArn": "arn:aws:sns:ap-south-1:ACCOUNT_ID:finetuning-notifications",
                "Subject": "Training Data Validation Failed",
                "Message": "Uploaded training data failed validation checks. Please review the data format."
            },
            "End": true
        },
        "NotifyFailure": {
            "Type": "Task",
            "Resource": "arn:aws:states:::sns:publish",
            "Parameters": {
                "TopicArn": "arn:aws:sns:ap-south-1:ACCOUNT_ID:finetuning-notifications",
                "Subject": "Fine-Tuning Pipeline Failed",
                "Message.$": "States.Format('Pipeline failed with error: {}', $.error.Cause)"
            },
            "End": true
        }
    }
}
```

#### Step 2 — Deploy the Automation Setup Script

Create `deploy_automation.py`:

```python
import boto3
import json
import time

# ─── CONFIGURATION ──────────────────────────────────────────────
REGION = "ap-south-1"
BUCKET_NAME = "bedrock-finetuning-your-unique-name"  # ← Change this!
ACCOUNT_ID = None  # We'll detect this automatically
# ─────────────────────────────────────────────────────────────────

def get_account_id():
    """Automatically detect the AWS account ID."""
    sts = boto3.client("sts")
    return sts.get_caller_identity()["Account"]

def create_eventbridge_rule():
    """Create an EventBridge rule that triggers when new training data is uploaded."""

    events = boto3.client("events", region_name=REGION)

    # This rule watches for new .jsonl files in our training-data folder
    event_pattern = {
        "source": ["aws.s3"],
        "detail-type": ["Object Created"],
        "detail": {
            "bucket": {"name": [BUCKET_NAME]},
            "object": {"key": [{"prefix": "training-data/"}]}
        }
    }

    print("  📡 Creating EventBridge rule...")

    response = events.put_rule(
        Name="bedrock-finetuning-trigger",
        Description="Triggers fine-tuning pipeline when new training data is uploaded to S3",
        EventPattern=json.dumps(event_pattern),
        State="ENABLED"
    )

    print(f"  ✅ EventBridge rule created: {response['RuleArn']}")
    return response["RuleArn"]

def enable_s3_event_notifications():
    """Enable S3 to send events to EventBridge."""

    s3 = boto3.client("s3", region_name=REGION)

    print("  🪣 Enabling S3 EventBridge notifications...")

    s3.put_bucket_notification_configuration(
        Bucket=BUCKET_NAME,
        NotificationConfiguration={
            "EventBridgeConfiguration": {}
        }
    )

    print("  ✅ S3 EventBridge notifications enabled")

def create_sns_topic():
    """Create an SNS topic for pipeline notifications."""

    sns = boto3.client("sns", region_name=REGION)

    print("  📧 Creating SNS notification topic...")

    response = sns.create_topic(Name="finetuning-notifications")
    topic_arn = response["TopicArn"]

    print(f"  ✅ SNS topic created: {topic_arn}")
    print(f"\n  📬 To receive notifications, subscribe your email:")
    print(f"     aws sns subscribe --topic-arn {topic_arn} \\")
    print(f"       --protocol email --notification-endpoint YOUR_EMAIL@example.com")

    return topic_arn

def deploy():
    """Deploy the complete automation infrastructure."""

    global ACCOUNT_ID

    print("=" * 60)
    print("  Bedrock Fine-Tuning — Automation Deployment")
    print("=" * 60)

    try:
        ACCOUNT_ID = get_account_id()
    except Exception as e:
        print(f"\n  ❌ AWS Connection Failed: {e}")
        print(f"\n  💡 Make sure you have configured AWS credentials:")
        print(f"     Run: aws configure")
        print("\n" + "=" * 60)
        return

    print(f"\n  🌏 Region      : {REGION}")
    print(f"  🪣 Bucket      : {BUCKET_NAME}")
    print(f"  🔢 Account ID  : {ACCOUNT_ID}")
    print()

    try:
        # Step 1: Enable S3 event notifications
        enable_s3_event_notifications()
        print()

        # Step 2: Create EventBridge rule
        rule_arn = create_eventbridge_rule()
        print()

        # Step 3: Create SNS notification topic
        topic_arn = create_sns_topic()
        print()
    except Exception as e:
        print(f"\n  ❌ Deployment Failed: {e}")
        print(f"\n  💡 Troubleshooting tips:")
        print(f"     1. Verify your S3 bucket exists: aws s3 ls s3://{BUCKET_NAME}")
        print(f"     2. Check your IAM permissions include EventBridge and SNS access")
        print(f"     3. Verify the region is correct: {REGION}")
        print("\n" + "=" * 60)
        return

    print("=" * 60)
    print("  ✅ Automation infrastructure deployed!")
    print("=" * 60)
    print(f"""
  What happens now:
  
  1. When you upload a new .jsonl file to:
     s3://{BUCKET_NAME}/training-data/
  
  2. EventBridge detects the upload and triggers
     the Step Functions pipeline
  
  3. The pipeline validates data → fine-tunes model →
     evaluates results → registers the model version
  
  4. You receive an email notification with the results

  To test it, upload a new training data file:
    aws s3 cp new_training_data.jsonl \\
      s3://{BUCKET_NAME}/training-data/new_training_data.jsonl
""")

    # Save deployment details
    deployment_info = {
        "account_id": ACCOUNT_ID,
        "region": REGION,
        "bucket": BUCKET_NAME,
        "eventbridge_rule_arn": rule_arn,
        "sns_topic_arn": topic_arn,
        "deployed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ")
    }

    with open("deployment_info.json", "w") as f:
        json.dump(deployment_info, f, indent=2)

    print("  💾 Deployment details saved to: deployment_info.json")

if __name__ == "__main__":
    deploy()
```

Run it:

```bash
python deploy_automation.py
```

#### What You Just Accomplished

You built the foundation of an **event-driven automation pipeline**. EventBridge is now watching your S3 bucket, and when new training data arrives, it can trigger the Step Functions state machine that orchestrates the entire fine-tuning process. This is the same architectural pattern used by production ML teams at major enterprises.

---

### Lab 4 — Compare Baseline vs Fine-Tuned Outputs

**Time:** ~1.5 hours  
**Goal:** Run the same prompts through both the original and fine-tuned models, then objectively compare  
**Prerequisite:** Lab 2 completed (fine-tuning job finished successfully)

This is the "moment of truth" lab — where you measure whether all that training actually made a difference.

#### The Evaluation Script

Create `evaluate_models.py`:

```python
import boto3
import json
from datetime import datetime

# ─── CONFIGURATION ──────────────────────────────────────────────
REGION = "ap-south-1"
BASE_MODEL_ID = "amazon.titan-text-express-v1"
# ─────────────────────────────────────────────────────────────────

def load_custom_model_name():
    """Load the custom model name from our saved job details."""
    try:
        with open("job_details.json", "r") as f:
            details = json.load(f)
            return details["custom_model_name"]
    except FileNotFoundError:
        print("  ❌ job_details.json not found. Run Lab 2 first.")
        exit(1)

def invoke_base_model(bedrock_runtime, prompt):
    """Send a prompt to the base (original) model."""
    body = json.dumps({
        "inputText": prompt,
        "textGenerationConfig": {
            "maxTokenCount": 300,
            "temperature": 0.1,
            "topP": 0.9
        }
    })

    try:
        response = bedrock_runtime.invoke_model(
            modelId=BASE_MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=body
        )

        result = json.loads(response["body"].read())
        return result["results"][0]["outputText"].strip()
    except Exception as e:
        return f"[Error invoking base model: {e}]"

def invoke_finetuned_model(bedrock_runtime, model_id, prompt):
    """Send a prompt to the fine-tuned model."""
    body = json.dumps({
        "inputText": prompt,
        "textGenerationConfig": {
            "maxTokenCount": 300,
            "temperature": 0.1,
            "topP": 0.9
        }
    })

    try:
        response = bedrock_runtime.invoke_model(
            modelId=model_id,
            contentType="application/json",
            accept="application/json",
            body=body
        )
        result = json.loads(response["body"].read())
        return result["results"][0]["outputText"].strip()
    except Exception as e:
        return f"[Error invoking fine-tuned model: {e}]"

def evaluate():
    """Compare base model vs fine-tuned model responses."""

    custom_model_name = load_custom_model_name()

    # Get the provisioned model ARN (required for custom models)
    bedrock = boto3.client("bedrock", region_name=REGION)
    bedrock_runtime = boto3.client("bedrock-runtime", region_name=REGION)

    print("=" * 65)
    print("  Model Evaluation — Baseline vs Fine-Tuned Comparison")
    print("=" * 65)
    print(f"\n  🧠 Base Model      : {BASE_MODEL_ID}")
    print(f"  🎯 Fine-Tuned Model: {custom_model_name}")

    # Test prompts — a mix of in-domain (banking) and out-of-domain queries
    test_prompts = [
        {
            "prompt": "What is the process to dispute a credit card transaction?",
            "category": "In-Domain (trained on similar example)",
            "expected_theme": "Steps for disputing transactions"
        },
        {
            "prompt": "How do I reset my online banking password?",
            "category": "In-Domain (related topic, not in training data)",
            "expected_theme": "Password reset process"
        },
        {
            "prompt": "What are the current mortgage rates?",
            "category": "In-Domain (banking, not in training data)",
            "expected_theme": "Current rate information"
        },
        {
            "prompt": "Can you explain how blockchain technology works?",
            "category": "Out-of-Domain (general tech question)",
            "expected_theme": "Technical explanation"
        }
    ]

    results = []

    for i, test in enumerate(test_prompts, 1):
        print(f"\n{'─' * 65}")
        print(f"  Test {i}/{len(test_prompts)}: {test['category']}")
        print(f"  Prompt: \"{test['prompt']}\"")
        print(f"{'─' * 65}")

        # Get base model response
        print(f"\n  🟦 BASE MODEL response:")
        base_response = invoke_base_model(bedrock_runtime, test["prompt"])
        print(f"  {base_response[:200]}{'...' if len(base_response) > 200 else ''}")

        # Get fine-tuned model response
        print(f"\n  🟩 FINE-TUNED MODEL response:")
        ft_response = invoke_finetuned_model(
            bedrock_runtime, custom_model_name, test["prompt"]
        )
        print(f"  {ft_response[:200]}{'...' if len(ft_response) > 200 else ''}")

        results.append({
            "prompt": test["prompt"],
            "category": test["category"],
            "base_response": base_response,
            "finetuned_response": ft_response,
            "expected_theme": test["expected_theme"]
        })

    # Save full results
    evaluation_report = {
        "evaluation_date": datetime.now().isoformat(),
        "base_model": BASE_MODEL_ID,
        "finetuned_model": custom_model_name,
        "results": results,
        "summary": {
            "total_tests": len(test_prompts),
            "in_domain_tests": sum(
                1 for t in test_prompts if "In-Domain" in t["category"]
            ),
            "out_of_domain_tests": sum(
                1 for t in test_prompts if "Out-of-Domain" in t["category"]
            )
        }
    }

    with open("evaluation_report.json", "w") as f:
        json.dump(evaluation_report, f, indent=2)

    print(f"\n{'=' * 65}")
    print(f"  📊 Evaluation Complete!")
    print(f"{'=' * 65}")
    print(f"\n  📁 Full report saved to: evaluation_report.json")
    print(f"\n  📈 What to look for:")
    print(f"     • In-Domain prompts: Fine-tuned model should give more")
    print(f"       specific, banking-appropriate responses")
    print(f"     • Out-of-Domain prompts: Both models should perform")
    print(f"       similarly (fine-tuning shouldn't hurt general ability)")
    print(f"     • Response format: Fine-tuned model may follow the")
    print(f"       structured format from our training examples")
    print(f"\n  💡 Tip: Open evaluation_report.json to read full responses")
    print(f"     side-by-side for detailed comparison.")

if __name__ == "__main__":
    evaluate()
```

Run it:

```bash
python evaluate_models.py
```

> **Note on Custom Model Inference:** To invoke a fine-tuned (custom) model in Bedrock, you may need to create a **Provisioned Throughput** endpoint first. Check the [Bedrock pricing page](https://aws.amazon.com/bedrock/pricing/) as this incurs additional costs. For learning purposes, you can also review the responses from the saved `evaluation_report.json` without provisioned throughput.

#### What You Just Accomplished

You built an **automated evaluation pipeline** that objectively compares your fine-tuned model against the baseline. This is a fundamental practice in ML engineering — you should never deploy a model without evidence that it's better than what you had before.

---

## Cleanup — Avoid Unexpected Charges

When you're done learning, clean up your resources to stop incurring AWS charges:

```bash
# Delete S3 training data and outputs
aws s3 rm s3://bedrock-finetuning-your-unique-name --recursive

# Delete the S3 bucket
aws s3 rb s3://bedrock-finetuning-your-unique-name

# Delete the IAM role and policy
aws iam delete-role-policy --role-name BedrockFineTuningRole --policy-name BedrockS3Access
aws iam delete-role --role-name BedrockFineTuningRole

# Delete EventBridge rule (if created in Lab 3)
aws events remove-targets --rule bedrock-finetuning-trigger --ids "1" 2>/dev/null
aws events delete-rule --name bedrock-finetuning-trigger 2>/dev/null

# Delete SNS topic (if created in Lab 3)
# First get the topic ARN
aws sns list-topics --query "Topics[?contains(TopicArn, 'finetuning-notifications')]" --output text
# Then delete it (replace with your actual topic ARN)
# aws sns delete-topic --topic-arn arn:aws:sns:ap-south-1:ACCOUNT_ID:finetuning-notifications

# Delete any provisioned throughput (if created for Lab 4)
# Check in Bedrock Console → Custom models → Provisioned throughput
```

> **Important:** Provisioned Throughput for custom models is the most expensive resource. Always delete it when you're done experimenting.

---

## Quick Reference — Key AWS CLI Commands

| Action | Command |
|---|---|
| List Bedrock models | `aws bedrock list-foundation-models --region ap-south-1` |
| Check fine-tuning jobs | `aws bedrock list-model-customization-jobs --region ap-south-1` |
| Get job status | `aws bedrock get-model-customization-job --job-identifier JOB_ARN` |
| List custom models | `aws bedrock list-custom-models --region ap-south-1` |
| List S3 buckets | `aws s3 ls` |
| List files in bucket | `aws s3 ls s3://your-bucket-name/` |
| Check IAM role | `aws iam get-role --role-name BedrockFineTuningRole` |

---

## Troubleshooting

### "AccessDeniedException" when starting fine-tuning
Your IAM role doesn't have the right permissions. Re-run the IAM setup commands in Lab 2, making sure the S3 bucket name in the policy matches your actual bucket.

### "ResourceNotFoundException" for the model
Model access hasn't been granted yet. Go to Bedrock Console → Model access and verify the Titan Text Express model shows "Access granted."

### Fine-tuning job stuck in "InProgress" for many hours
Jobs can take 1–4+ hours depending on dataset size. If it's been over 6 hours with no progress, check the job details for error messages.

### "ValidationException" on training data
Your JSONL file has formatting issues. Common fixes: ensure every line is valid JSON with exactly `prompt` and `completion` fields, no trailing commas, and no empty lines.

### EventBridge rule not triggering
Verify S3 EventBridge notifications are enabled on your bucket. Run the `enable_s3_event_notifications()` function from Lab 3 again.

---

## What's Next?

Once you're comfortable with these labs, explore:

- **Advanced Evaluation** — Use Claude to auto-score fine-tuned responses on relevance, accuracy, and safety
- **A/B Testing** — Route production traffic between baseline and fine-tuned models to measure real-world impact
- **Continuous Training** — Set up weekly automated retraining with fresh customer interaction data
- **Multi-Model Strategy** — Fine-tune different models for different use cases (e.g., one for customer support, one for document summarisation)
- **Guardrails Integration** — Add Bedrock Guardrails to ensure fine-tuned models remain safe and compliant

---

## Resources

- [Amazon Bedrock Documentation](https://docs.aws.amazon.com/bedrock/)
- [Bedrock Custom Models Guide](https://docs.aws.amazon.com/bedrock/latest/userguide/custom-models.html)
- [AWS Step Functions Developer Guide](https://docs.aws.amazon.com/step-functions/latest/dg/)
- [Amazon EventBridge User Guide](https://docs.aws.amazon.com/eventbridge/latest/userguide/)
- [Bedrock Pricing](https://aws.amazon.com/bedrock/pricing/)

---

> **Built for learners.** If something in this guide was confusing or didn't work, that's our problem to fix — not yours. Open an issue and let us know.
