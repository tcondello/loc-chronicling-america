# AWS CDK Deployment: Chronicling America Nationwide Ingestion Pipeline

This directory contains the production AWS CDK infrastructure stack (`ChronAmPipelineStack`) to provision a dedicated, self-healing EC2 worker that ingests historical newspapers across all 50 US states and pushes them directly to Hugging Face.

---

## Architecture Overview

```text
AWS Cloud (Single-AZ VPC - $0 NAT Gateway)
├── EC2 Instance (c6i.xlarge, Ubuntu 24.04 LTS, 100 GB GP3 EBS)
│   ├── UserData: Auto-bootstrap repo, venv, and dependencies
│   ├── Systemd Service (loc-pipeline.service): Auto-restarts on any failure
│   └── Amazon CloudWatch Agent: Streams logs to /aws/ec2/loc-chronicling-america
├── IAM Role: AWS Systems Manager (SSM) keyless terminal access + CloudWatch
└── Live Destination: https://huggingface.co/datasets/Tim-Pinecone/LOC-Chronicling-America
```

---

## Quickstart Deployment

### 1. Install Prerequisites
```bash
npm install -g aws-cdk
pip install -r requirements.txt
```

### 2. Configure AWS Credentials
Ensure your AWS CLI session is active:
```bash
# Standard access keys:
aws configure

# Or AWS IAM Identity Center (SSO):
aws sso login
```

Verify your credentials:
```bash
aws sts get-caller-identity
```

### 3. Bootstrap CDK (First time in account/region)
```bash
cdk bootstrap
```

### 4. Deploy Stack
Deploy with your Hugging Face write token:

```bash
cdk deploy -c hf_token=<YOUR_HF_TOKEN>
```

#### Optional Context Overrides:
```bash
cdk deploy \
  -c hf_token=<YOUR_HF_TOKEN> \
  -c region=us-east-1 \
  -c instance_type=c6i.xlarge \
  -c volume_size_gb=100
```

---

## Monitoring the Pipeline

Once deployed, the stack automatically starts running the nationwide pipeline via `systemd`.

### Stream Live CloudWatch Logs
```bash
aws logs tail /aws/ec2/loc-chronicling-america --follow
```

### Connect to the Worker Terminal (Keyless via SSM)
No SSH keys or open inbound ports are needed. Use AWS Systems Manager:

```bash
aws ssm start-session --target <InstanceId-from-CDK-Output>
```

Once connected to the worker, you can check:
```bash
# Check service status
systemctl status loc-pipeline.service

# Live tail local log file
tail -f /var/log/chronam-pipeline.log

# Inspect system resource utilization
htop
df -h
```

---

## Teardown / Cleanup
Once ingestion is completed, destroy the AWS resources:
```bash
cdk destroy
```
