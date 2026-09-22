# AWS CDK & EC2 Deployment Engineering Rules

These rules govern all AWS CDK infrastructure, EC2 UserData, cloud-init scripts, and remote terminal interactions in this repository.

## 1. CloudFormation EC2 UserData & Instance Replacement
* **UserData Never Re-Runs In-Place**: Updating the `UserData` property on an `AWS::EC2::Instance` in CloudFormation is classified as "No interruption". CloudFormation will update the metadata on the existing instance WITHOUT rebooting or replacing it.
* **Enforce Replacement When UserData Changes**:
  - In CDK stacks using raw `ec2.Instance`, change the logical ID or construct ID (e.g. `ChronAmWorkerV3`) when changing `UserData` so CloudFormation terminates the stale instance and provisions a fresh one.
  - Alternatively, use an `ec2.LaunchTemplate` with an Auto Scaling Group (min=1, max=1) configured with instance refresh.

## 2. OS-Specific Package & Daemon Invariants
* **Ubuntu Amazon CloudWatch Agent**: Never attempt `apt-get install -y amazon-cloudwatch-agent` on Debian/Ubuntu. It is not available in default repositories. Always download and install the official Debian package:
  ```bash
  curl -fsSL https://s3.amazonaws.com/amazoncloudwatch-agent/ubuntu/amd64/latest/amazon-cloudwatch-agent.deb -o /tmp/amazon-cloudwatch-agent.deb
  dpkg -i -E /tmp/amazon-cloudwatch-agent.deb
  rm -f /tmp/amazon-cloudwatch-agent.deb
  ```
* **Ubuntu 24.04+ Python Virtual Environments**: Always install `python3.12-venv` (or release-specific venv package) alongside `python3-pip` and `python3-venv`. Without it, `python3 -m venv` fails with `ensurepip is not available`.

## 3. Remote Shell & SSM Automation Guidelines
* **Zero Heredocs in Interactive Sessions**: Never ask the user to copy-paste multi-line scripts containing `cat << 'EOF'` heredocs into remote interactive sessions (e.g. AWS Systems Manager / SSH). Remote PTY buffers frequently swallow newlines and mangle multi-line input.
* **Version-Controlled Setup Scripts**: Always commit the setup logic to an executable script in the repository (e.g. `scripts/setup_worker.sh`). Have the remote machine run `git pull && bash scripts/setup_worker.sh <args>`.
* **Git Multi-User Safety**: When scripts run as `root` (or under `sudo`) in an application directory owned by another user (e.g. `/home/ubuntu`), always configure safe directory ownership first:
  ```bash
  git config --global --add safe.directory "$APP_DIR"
  ```

## 4. CDK v2 Configuration Standards
* **cdk.json**: Ensure `cdk.json` does not contain deprecated CDK v1 flags (e.g. `@aws-cdk/core:enableStackNameDuplicates`, `aws-cdk:enableDiffNoFail`).
* **GP3 Volumes on EC2 Instances**: Do not set `throughput` or `iops` parameters directly on `ec2.BlockDeviceVolume.ebs()` for `AWS::EC2::Instance`. Baseline GP3 (3,000 IOPS / 125 MB/s) is automatically provided at $0 cost; CloudFormation only accepts explicit throughput parameters on Launch Templates.
* **Environment Resolution**: In `app.py`, resolve `account` from context or environment variables, and pass `env = cdk.Environment(account=account, region=region) if account else None` so offline `cdk synth` succeeds without requiring active AWS CLI credentials.
