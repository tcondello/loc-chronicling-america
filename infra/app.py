#!/usr/bin/env python3
"""AWS CDK App entrypoint for Chronicling America ingestion pipeline."""

import os
import aws_cdk as cdk
from pipeline_stack import ChronAmPipelineStack

app = cdk.App()

# Read settings from CDK context or environment variables
hf_token = app.node.try_get_context("hf_token") or os.environ.get("HF_TOKEN")
if not hf_token:
    raise ValueError("HF_TOKEN must be specified via CDK context (-c hf_token=...) or environment variable.")

hf_repo = app.node.try_get_context("hf_repo") or os.environ.get("HF_REPO", "Tim-Pinecone/LOC-Chronicling-America")
instance_type = app.node.try_get_context("instance_type") or "c6i.xlarge"
volume_size_gb = int(app.node.try_get_context("volume_size_gb") or 100)

env = cdk.Environment(
    account=os.environ.get("CDK_DEFAULT_ACCOUNT"),
    region=os.environ.get("CDK_DEFAULT_REGION", "us-east-1"),
)

ChronAmPipelineStack(
    app,
    "ChronAmPipelineStack",
    hf_token=hf_token,
    hf_repo=hf_repo,
    instance_type=instance_type,
    volume_size_gb=volume_size_gb,
    env=env,
    description="Automated, non-stop Library of Congress Chronicling America streaming ingestion pipeline to Hugging Face",
)

app.synth()
