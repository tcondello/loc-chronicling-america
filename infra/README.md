# AWS CDK Deployment: Chronicling America Nationwide Ingestion Pipeline

This directory contains the production AWS CDK infrastructure stack (`ChronAmPipelineStack`) to provision a dedicated, self-healing EC2 worker that ingests historical newspapers across all 50 US states and pushes them directly to Hugging Face.

---

## Architecture Overview

```
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
Ensure your AWS CLI credentials are set:
```bash
export AWS_REGION=us-east-1
# Or run: aws configure
```

### 3. Deploy Stack
```bash
# Pass token via environment variable or context parameter
export HF_TOKEN="<YOUR_HF_TOKEN>"
cdk deploy -c hf_token=$HF_TOKEN
```

---

## Monitoring the Pipeline

Once deployed, the stack automatically starts running the nationwide pipeline via `systemd`.

### Stream Live CloudWatch Logs:
```bash
aws logs tail /aws/ec2/loc-chronicling-america --follow
```

### Connect to the Worker Terminal (Keyless via SSM):
```bash
aws ssm start-session --target <InstanceId-from-CDK-Output>
```

Once connected to the worker, you can check:
```bash
# Check service status
systemctl status loc-pipeline.service

# Live tail local log file
tail -f /var/log/chronam-pipeline.log

# System resource utilization
htop
df -h
```

---

## Teardown / Cleanup
Once all 50 states are completed, destroy the AWS resources:
```bash
cdk destroy
```
