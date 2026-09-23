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

account = (
    app.node.try_get_context("account")
    or os.environ.get("CDK_DEFAULT_ACCOUNT")
    or os.environ.get("AWS_ACCOUNT_ID")
)
region = (
    app.node.try_get_context("region")
    or os.environ.get("CDK_DEFAULT_REGION")
    or os.environ.get("AWS_REGION")
    or os.environ.get("AWS_DEFAULT_REGION")
    or "us-east-1"
)

# Only construct Environment if account is explicitly resolved;
# otherwise pass env=None for an environment-agnostic stack.
env = cdk.Environment(account=account, region=region) if account else None

states = app.node.try_get_context("states")

ChronAmPipelineStack(
    app,
    "ChronAmPipelineStack",
    hf_token=hf_token,
    hf_repo=hf_repo,
    instance_type=instance_type,
    volume_size_gb=volume_size_gb,
    states=states,
    env=env,
    description="Automated, non-stop Library of Congress Chronicling America streaming ingestion pipeline to Hugging Face",
)

app.synth()
