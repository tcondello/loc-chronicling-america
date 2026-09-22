"""AWS CDK Stack for automated, non-stop Chronicling America ingestion pipeline."""

from typing import Optional
import aws_cdk as cdk
from aws_cdk import (
    Stack,
    CfnOutput,
    RemovalPolicy,
    aws_ec2 as ec2,
    aws_iam as iam,
    aws_logs as logs,
)
from constructs import Construct


class ChronAmPipelineStack(Stack):
    """Provisions a self-healing, high-throughput EC2 worker to ingest Chronicling America."""

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        hf_token: str,
        hf_repo: str = "Tim-Pinecone/LOC-Chronicling-America",
        instance_type: str = "c6i.xlarge",
        volume_size_gb: int = 100,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # 1. Cost-Optimized VPC (1 AZ, Public Subnet, $0 NAT Gateway)
        vpc = ec2.Vpc(
            self,
            "ChronAmVpc",
            max_azs=1,
            nat_gateways=0,
            subnet_configuration=[
                ec2.SubnetConfiguration(
                    name="Public",
                    subnet_type=ec2.SubnetType.PUBLIC,
                    cidr_mask=24,
                )
            ],
        )

        # 2. CloudWatch Log Group for streaming pipeline logs
        log_group = logs.LogGroup(
            self,
            "ChronAmLogGroup",
            log_group_name="/aws/ec2/loc-chronicling-america",
            retention=logs.RetentionDays.ONE_MONTH,
            removal_policy=RemovalPolicy.DESTROY,
        )

        # 3. Security Group (Outbound only - access via AWS SSM Session Manager, no open SSH ports needed)
        security_group = ec2.SecurityGroup(
            self,
            "ChronAmSecurityGroup",
            vpc=vpc,
            description="Security group for Chronicling America ingestion worker",
            allow_all_outbound=True,
        )

        # 4. IAM Role for Keyless SSM Access & CloudWatch Logging
        role = iam.Role(
            self,
            "ChronAmInstanceRole",
            assumed_by=iam.ServicePrincipal("ec2.amazonaws.com"),
            description="Role allowing SSM terminal access and CloudWatch log streaming",
        )
        role.add_managed_policy(
            iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSSMManagedInstanceCore")
        )
        role.add_managed_policy(
            iam.ManagedPolicy.from_aws_managed_policy_name("CloudWatchAgentServerPolicy")
        )

        # 5. Ubuntu 24.04 LTS AMI (x86_64)
        ubuntu_ami = ec2.MachineImage.from_ssm_parameter(
            "/aws/service/canonical/ubuntu/server/24.04/stable/current/amd64/hvm/ebs-gp3/ami-id"
        )

        # 6. UserData: Bootstrap, Clone, Virtualenv, and Auto-Restarting Systemd Service
        user_data = ec2.UserData.for_linux()
        user_data.add_commands(
            "#!/usr/bin/env bash",
            "set -euxo pipefail",
            "",
            "# 1. Configure 4 GB Swap for memory headroom",
            "if [ ! -f /swapfile ]; then",
            "    fallocate -l 4G /swapfile",
            "    chmod 600 /swapfile",
            "    mkswap /swapfile",
            "    swapon /swapfile",
            "    echo '/swapfile none swap sw 0 0' >> /etc/fstab",
            "fi",
            "",
            "# 2. Install system packages and CloudWatch agent",
            "export DEBIAN_FRONTEND=noninteractive",
            "apt-get update -y",
            "apt-get install -y git python3 python3-pip python3-venv python3.12-venv tmux curl wget bzip2 htop",
            "curl -fsSL https://s3.amazonaws.com/amazoncloudwatch-agent/ubuntu/amd64/latest/amazon-cloudwatch-agent.deb -o /tmp/amazon-cloudwatch-agent.deb",
            "dpkg -i -E /tmp/amazon-cloudwatch-agent.deb",
            "rm -f /tmp/amazon-cloudwatch-agent.deb",
            "",
            "# 3. Clone Repository into ubuntu user home",
            "USER_HOME=/home/ubuntu",
            "APP_DIR=${USER_HOME}/loc-chronicling-america",
            "if [ ! -d \"${APP_DIR}\" ]; then",
            "    git clone https://github.com/tcondello/loc-chronicling-america.git \"${APP_DIR}\"",
            "    chown -R ubuntu:ubuntu \"${APP_DIR}\"",
            "fi",
            "",
            "# 4. Build Virtual Environment",
            "rm -rf \"${APP_DIR}/.venv\"",
            "sudo -u ubuntu python3 -m venv \"${APP_DIR}/.venv\"",
            "sudo -u ubuntu \"${APP_DIR}/.venv/bin/pip\" install --upgrade pip",
            "sudo -u ubuntu bash -c \"cd ${APP_DIR} && .venv/bin/pip install -e '.[pipeline]'\"",
            "",
            f"# 5. Injected Hugging Face Credentials & Configuration",
            f"cat << 'EOF' > \"${{APP_DIR}}/.env\"",
            f"HF_TOKEN={hf_token}",
            f"HF_REPO={hf_repo}",
            "PYTHONUNBUFFERED=1",
            "EOF",
            "chown ubuntu:ubuntu \"${APP_DIR}/.env\"",
            "chmod 600 \"${APP_DIR}/.env\"",
            "",
            "# 6. Configure CloudWatch Agent to tail pipeline logs",
            "touch /var/log/chronam-pipeline.log",
            "chown ubuntu:ubuntu /var/log/chronam-pipeline.log",
            "mkdir -p /opt/aws/amazon-cloudwatch-agent/etc",
            "cat << 'EOF' > /opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json",
            "{",
            '  "logs": {',
            '    "logs_collected": {',
            '      "files": {',
            '        "collect_list": [',
            '          {',
            '            "file_path": "/var/log/chronam-pipeline.log",',
            '            "log_group_name": "/aws/ec2/loc-chronicling-america",',
            '            "log_stream_name": "{instance_id}/pipeline.log",',
            '            "timezone": "UTC"',
            '          }',
            '        ]',
            '      }',
            '    }',
            '  }',
            "}",
            "EOF",
            "/opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl -a fetch-config -m ec2 -s -c file:/opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json",
            "",
            "# 7. Create Systemd Service for Unattended Execution with Auto-Restart",
            "cat << 'EOF' > /etc/systemd/system/loc-pipeline.service",
            "[Unit]",
            "Description=Library of Congress Chronicling America Nationwide Streaming Pipeline",
            "After=network-online.target",
            "Wants=network-online.target",
            "",
            "[Service]",
            "Type=simple",
            "User=ubuntu",
            "WorkingDirectory=/home/ubuntu/loc-chronicling-america",
            "EnvironmentFile=/home/ubuntu/loc-chronicling-america/.env",
            "ExecStart=/home/ubuntu/loc-chronicling-america/.venv/bin/python -u scripts/run_multi_state_pipeline.py --all-states --purge-local-after-upload",
            "Restart=always",
            "RestartSec=15",
            "StandardOutput=append:/var/log/chronam-pipeline.log",
            "StandardError=append:/var/log/chronam-pipeline.log",
            "",
            "[Install]",
            "WantedBy=multi-user.target",
            "EOF",
            "",
            "systemctl daemon-reload",
            "systemctl enable loc-pipeline.service",
            "systemctl start loc-pipeline.service",
        )

        # 7. EC2 Worker Instance (High Network Throughput & NVMe/GP3 SSD)
        instance = ec2.Instance(
            self,
            "ChronAmWorkerV2",
            instance_type=ec2.InstanceType(instance_type),
            machine_image=ubuntu_ami,
            vpc=vpc,
            vpc_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PUBLIC),
            security_group=security_group,
            role=role,
            user_data=user_data,
            block_devices=[
                ec2.BlockDevice(
                    device_name="/dev/sda1",
                    volume=ec2.BlockDeviceVolume.ebs(
                        volume_size=volume_size_gb,
                        volume_type=ec2.EbsDeviceVolumeType.GP3,
                    ),
                )
            ],
        )

        # 8. Outputs for Easy Monitoring
        CfnOutput(
            self,
            "SSMConnectCommand",
            value=f"aws ssm start-session --target {instance.instance_id}",
            description="Run this command to open a shell into the worker without SSH keys",
        )
        CfnOutput(
            self,
            "CloudWatchLogsCommand",
            value=f"aws logs tail /aws/ec2/loc-chronicling-america --follow",
            description="Run this command to live-stream pipeline logs to your terminal",
        )
        CfnOutput(
            self,
            "HuggingFaceDatasetURL",
            value=f"https://huggingface.co/datasets/{hf_repo}",
            description="Live dataset destination on Hugging Face",
        )
        CfnOutput(
            self,
            "WorkerInstanceId",
            value=instance.instance_id,
            description="EC2 Instance ID",
        )
