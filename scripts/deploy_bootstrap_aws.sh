#!/bin/bash
# =============================================================================
# PRSM Bootstrap Server AWS Deployment Script
# =============================================================================
# This script deploys the PRSM bootstrap server to AWS infrastructure.
# 
# Prerequisites:
#   - AWS CLI installed and configured
#   - Docker installed
#   - jq installed
#   - Valid SSL certificates in /etc/ssl/prsm/
#
# Usage:
#   ./deploy_bootstrap_aws.sh [options]
#
# Options:
#   -r, --region       AWS region (default: us-east-1)
#   -e, --environment  Environment (staging/production)
#   -d, --domain       Domain name for the bootstrap server
#   -h, --help         Show this help message
#
# Environment Variables:
#   AWS_REGION          AWS region
#   PRSM_DOMAIN         Domain name
#   PRSM_ENV            Environment name
#   POSTGRES_PASSWORD   PostgreSQL password
#   GRAFANA_ADMIN_PASSWORD  Grafana admin password
# =============================================================================

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================

# Default values
DEFAULT_REGION="us-east-1"
DEFAULT_ENVIRONMENT="production"
DEFAULT_DOMAIN="prsm-network.com"
DEFAULT_INSTANCE_TYPE="t3.medium"
DEFAULT_KEY_NAME="prsm-bootstrap-key"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# =============================================================================
# Helper Functions
# =============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

show_help() {
    cat << EOF
PRSM Bootstrap Server AWS Deployment Script

Usage: $0 [options]

Options:
    -r, --region       AWS region (default: $DEFAULT_REGION)
    -e, --environment  Environment: staging/production (default: $DEFAULT_ENVIRONMENT)
    -d, --domain       Domain name (default: $DEFAULT_DOMAIN)
    -i, --instance-type  EC2 instance type (default: $DEFAULT_INSTANCE_TYPE)
    -k, --key-name     EC2 key pair name (default: $DEFAULT_KEY_NAME)
    -h, --help         Show this help message

Environment Variables:
    AWS_REGION              AWS region
    PRSM_DOMAIN             Domain name
    PRSM_ENV                Environment name
    POSTGRES_PASSWORD       PostgreSQL password
    GRAFANA_ADMIN_PASSWORD  Grafana admin password
    PRSM_AUTH_SECRET        Authentication secret for bootstrap server

Examples:
    $0 --region us-west-2 --environment staging
    $0 -r eu-west-1 -e production -d prsm-network.com

EOF
    exit 0
}

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    local missing=()
    
    # Check AWS CLI
    if ! command -v aws &> /dev/null; then
        missing+=("aws-cli")
    fi
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        missing+=("docker")
    fi
    
    # Check jq
    if ! command -v jq &> /dev/null; then
        missing+=("jq")
    fi
    
    if [ ${#missing[@]} -gt 0 ]; then
        log_error "Missing required tools: ${missing[*]}"
        log_error "Please install them before running this script."
        exit 1
    fi
    
    # Check AWS credentials
    if ! aws sts get-caller-identity &> /dev/null; then
        log_error "AWS credentials not configured. Run 'aws configure' first."
        exit 1
    fi
    
    log_success "All prerequisites met"
}

# =============================================================================
# AWS Resource Functions
# =============================================================================

create_vpc() {
    local vpc_name="prsm-bootstrap-${ENVIRONMENT}-vpc"
    
    log_info "Creating VPC: $vpc_name"
    
    # Check if VPC already exists
    local vpc_id=$(aws ec2 describe-vpcs \
        --filters "Name=tag:Name,Values=$vpc_name" \
        --query "Vpcs[0].VpcId" \
        --output text \
        --region "$AWS_REGION" 2>/dev/null || echo "None")
    
    if [ "$vpc_id" != "None" ] && [ "$vpc_id" != "" ]; then
        log_warning "VPC already exists: $vpc_id"
        echo "$vpc_id"
        return 0
    fi
    
    # Create VPC
    vpc_id=$(aws ec2 create-vpc \
        --cidr-block 10.0.0.0/16 \
        --tag-specifications "ResourceType=vpc,Tags=[{Key=Name,Value=$vpc_name},{Key=Environment,Value=$ENVIRONMENT},{Key=Project,Value=PRSM}]" \
        --query "Vpc.VpcId" \
        --output text \
        --region "$AWS_REGION")
    
    # Enable DNS hostnames
    aws ec2 modify-vpc-attribute \
        --vpc-id "$vpc_id" \
        --enable-dns-hostnames \
        --region "$AWS_REGION"
    
    log_success "Created VPC: $vpc_id"
    echo "$vpc_id"
}

create_subnet() {
    local vpc_id=$1
    local subnet_name="prsm-bootstrap-${ENVIRONMENT}-subnet"
    
    log_info "Creating subnet: $subnet_name"
    
    # Check if subnet already exists
    local subnet_id=$(aws ec2 describe-subnets \
        --filters "Name=tag:Name,Values=$subnet_name" "Name=vpc-id,Values=$vpc_id" \
        --query "Subnets[0].SubnetId" \
        --output text \
        --region "$AWS_REGION" 2>/dev/null || echo "None")
    
    if [ "$subnet_id" != "None" ] && [ "$subnet_id" != "" ]; then
        log_warning "Subnet already exists: $subnet_id"
        echo "$subnet_id"
        return 0
    fi
    
    # Get availability zone
    local az=$(aws ec2 describe-availability-zones \
        --region "$AWS_REGION" \
        --query "AvailabilityZones[0].ZoneName" \
        --output text)
    
    # Create subnet
    subnet_id=$(aws ec2 create-subnet \
        --vpc-id "$vpc_id" \
        --cidr-block 10.0.1.0/24 \
        --availability-zone "$az" \
        --tag-specifications "ResourceType=subnet,Tags=[{Key=Name,Value=$subnet_name},{Key=Environment,Value=$ENVIRONMENT}]" \
        --query "Subnet.SubnetId" \
        --output text \
        --region "$AWS_REGION")
    
    # Enable auto-assign public IP
    aws ec2 modify-subnet-attribute \
        --subnet-id "$subnet_id" \
        --map-public-ip-on-launch \
        --region "$AWS_REGION"
    
    log_success "Created subnet: $subnet_id"
    echo "$subnet_id"
}

create_internet_gateway() {
    local vpc_id=$1
    local igw_name="prsm-bootstrap-${ENVIRONMENT}-igw"
    
    log_info "Creating Internet Gateway: $igw_name"
    
    # Check if IGW already exists
    local igw_id=$(aws ec2 describe-internet-gateways \
        --filters "Name=tag:Name,Values=$igw_name" \
        --query "InternetGateways[0].InternetGatewayId" \
        --output text \
        --region "$AWS_REGION" 2>/dev/null || echo "None")
    
    if [ "$igw_id" != "None" ] && [ "$igw_id" != "" ]; then
        log_warning "Internet Gateway already exists: $igw_id"
        echo "$igw_id"
        return 0
    fi
    
    # Create IGW
    igw_id=$(aws ec2 create-internet-gateway \
        --tag-specifications "ResourceType=internet-gateway,Tags=[{Key=Name,Value=$igw_name}]" \
        --query "InternetGateway.InternetGatewayId" \
        --output text \
        --region "$AWS_REGION")
    
    # Attach to VPC
    aws ec2 attach-internet-gateway \
        --internet-gateway-id "$igw_id" \
        --vpc-id "$vpc_id" \
        --region "$AWS_REGION"
    
    log_success "Created Internet Gateway: $igw_id"
    echo "$igw_id"
}

create_route_table() {
    local vpc_id=$1
    local igw_id=$2
    local subnet_id=$3
    local rt_name="prsm-bootstrap-${ENVIRONMENT}-rt"
    
    log_info "Creating Route Table: $rt_name"
    
    # Check if route table already exists
    local rt_id=$(aws ec2 describe-route-tables \
        --filters "Name=tag:Name,Values=$rt_name" "Name=vpc-id,Values=$vpc_id" \
        --query "RouteTables[0].RouteTableId" \
        --output text \
        --region "$AWS_REGION" 2>/dev/null || echo "None")
    
    if [ "$rt_id" != "None" ] && [ "$rt_id" != "" ]; then
        log_warning "Route Table already exists: $rt_id"
        echo "$rt_id"
        return 0
    fi
    
    # Create route table
    rt_id=$(aws ec2 create-route-table \
        --vpc-id "$vpc_id" \
        --tag-specifications "ResourceType=route-table,Tags=[{Key=Name,Value=$rt_name}]" \
        --query "RouteTable.RouteTableId" \
        --output text \
        --region "$AWS_REGION")
    
    # Add route to Internet Gateway
    aws ec2 create-route \
        --route-table-id "$rt_id" \
        --destination-cidr-block 0.0.0.0/0 \
        --gateway-id "$igw_id" \
        --region "$AWS_REGION"
    
    # Associate with subnet
    aws ec2 associate-route-table \
        --route-table-id "$rt_id" \
        --subnet-id "$subnet_id" \
        --region "$AWS_REGION"
    
    log_success "Created Route Table: $rt_id"
    echo "$rt_id"
}

create_security_group() {
    local vpc_id=$1
    local sg_name="prsm-bootstrap-${ENVIRONMENT}-sg"
    
    log_info "Creating Security Group: $sg_name"
    
    # Check if SG already exists
    local sg_id=$(aws ec2 describe-security-groups \
        --filters "Name=group-name,Values=$sg_name" "Name=vpc-id,Values=$vpc_id" \
        --query "SecurityGroups[0].GroupId" \
        --output text \
        --region "$AWS_REGION" 2>/dev/null || echo "None")
    
    if [ "$sg_id" != "None" ] && [ "$sg_id" != "" ]; then
        log_warning "Security Group already exists: $sg_id"
        echo "$sg_id"
        return 0
    fi
    
    # Create SG
    sg_id=$(aws ec2 create-security-group \
        --group-name "$sg_name" \
        --description "PRSM Bootstrap Server Security Group" \
        --vpc-id "$vpc_id" \
        --query "GroupId" \
        --output text \
        --region "$AWS_REGION")
    
    # Add inbound rules
    # SSH
    aws ec2 authorize-security-group-ingress \
        --group-id "$sg_id" \
        --protocol tcp \
        --port 22 \
        --cidr 0.0.0.0/0 \
        --region "$AWS_REGION"
    
    # WebSocket P2P
    aws ec2 authorize-security-group-ingress \
        --group-id "$sg_id" \
        --protocol tcp \
        --port 8765 \
        --cidr 0.0.0.0/0 \
        --region "$AWS_REGION"
    
    # HTTP API
    aws ec2 authorize-security-group-ingress \
        --group-id "$sg_id" \
        --protocol tcp \
        --port 8000 \
        --cidr 0.0.0.0/0 \
        --region "$AWS_REGION"
    
    # HTTPS
    aws ec2 authorize-security-group-ingress \
        --group-id "$sg_id" \
        --protocol tcp \
        --port 443 \
        --cidr 0.0.0.0/0 \
        --region "$AWS_REGION"
    
    # Prometheus metrics (internal only)
    aws ec2 authorize-security-group-ingress \
        --group-id "$sg_id" \
        --protocol tcp \
        --port 9090 \
        --cidr 10.0.0.0/16 \
        --region "$AWS_REGION"
    
    log_success "Created Security Group: $sg_id"
    echo "$sg_id"
}

launch_ec2_instance() {
    local subnet_id=$1
    local sg_id=$2
    
    log_info "Launching EC2 instance..."
    
    # Get latest Amazon Linux 2 AMI
    local ami_id=$(aws ec2 describe-images \
        --owners amazon \
        --filters "Name=name,Values=amzn2-ami-hvm-*-x86_64-gp2" "Name=state,Values=available" \
        --query "Images | sort_by(@, &CreationDate) | [-1].ImageId" \
        --output text \
        --region "$AWS_REGION")
    
    log_info "Using AMI: $ami_id"
    
    # Create user data script
    local user_data=$(cat << 'USERDATA'
#!/bin/bash
# PRSM Bootstrap Server Instance Setup

# Update system
yum update -y

# Install Docker
amazon-linux-extras install docker -y
systemctl start docker
systemctl enable docker
usermod -a -G docker ec2-user

# Install Docker Compose (sp1247: pin the version + verify an out-of-band SHA256
# before making the binary executable — supply-chain audit #7. The previous line
# pulled from a floating releases/latest URL with NO integrity check straight into
# /usr/local/bin and chmod +x'd it as root, so a MITM or a compromised release asset
# would run arbitrary root code on every bootstrap node. A checksum fetched from the
# same release can't protect against a compromised release, so the expected digests
# are hardcoded here and pinned to a specific version. The old URL also built the v1
# asset name docker-compose-Linux-x86_64, which 404s against v2+'s lowercase
# docker-compose-linux-x86_64 — this also fixes that latent break.)
# NOTE: an if/elif ladder is used rather than `case` on purpose — this whole block
# is captured inside a `$(cat << 'USERDATA' ... )` command substitution, and `case`
# pattern terminators (`x86_64)`) are bare `)` that the outer $() parser miscounts as
# closing the substitution. if/elif keeps every paren balanced.
DOCKER_COMPOSE_VERSION="v5.2.0"
DC_ARCH_M="$(uname -m)"
if [ "$DC_ARCH_M" = "x86_64" ]; then
  DC_ARCH="x86_64";  DC_SHA256="018f9612ecabc5f2d7aaa53d6f5f44453a87611e2d72c8ef84d7b1eca070e719"
elif [ "$DC_ARCH_M" = "aarch64" ] || [ "$DC_ARCH_M" = "arm64" ]; then
  DC_ARCH="aarch64"; DC_SHA256="739de570a0adf5eab12830db980f549fb5f44ad6b266e1e43e20f6f9df7cbcca"
else
  echo "FATAL: unsupported arch $DC_ARCH_M for the docker-compose checksum pin" >&2
  exit 1
fi
DC_URL="https://github.com/docker/compose/releases/download/${DOCKER_COMPOSE_VERSION}/docker-compose-linux-${DC_ARCH}"
curl -fSL "$DC_URL" -o /tmp/docker-compose
echo "${DC_SHA256}  /tmp/docker-compose" | sha256sum -c - \
  || { echo "FATAL: docker-compose checksum mismatch — refusing to install" >&2; rm -f /tmp/docker-compose; exit 1; }
install -m 0755 /tmp/docker-compose /usr/local/bin/docker-compose
rm -f /tmp/docker-compose

# Install AWS CLI (already installed on Amazon Linux 2)
# Install jq
yum install -y jq

# Create PRSM directory
mkdir -p /opt/prsm
cd /opt/prsm

# Create environment file
cat > .env << EOF
PRSM_ENV=production
PRSM_DOMAIN=${PRSM_DOMAIN}
PRSM_REGION=${AWS_REGION}
POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
GRAFANA_ADMIN_PASSWORD=${GRAFANA_ADMIN_PASSWORD}
PRSM_AUTH_SECRET=${PRSM_AUTH_SECRET}
EOF

# Pull and run bootstrap server
docker-compose -f docker-compose.bootstrap.yml up -d

# Setup log rotation
cat > /etc/logrotate.d/prsm << EOF
/var/lib/docker/containers/*/*.log {
    rotate 7
    daily
    compress
    size=100M
    missingok
    delaycompress
    copytruncate
}
EOF

USERDATA
)
    
    # Launch instance
    local instance_id=$(aws ec2 run-instances \
        --image-id "$ami_id" \
        --count 1 \
        --instance-type "$INSTANCE_TYPE" \
        --key-name "$KEY_NAME" \
        --subnet-id "$subnet_id" \
        --security-group-ids "$sg_id" \
        --user-data "$user_data" \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=prsm-bootstrap-${ENVIRONMENT}},{Key=Environment,Value=$ENVIRONMENT},{Key=Project,Value=PRSM}]" \
        --block-device-mappings "[{\"DeviceName\":\"/dev/xvda\",\"Ebs\":{\"VolumeSize\":50,\"VolumeType\":\"gp3\"}}]" \
        --query "Instances[0].InstanceId" \
        --output text \
        --region "$AWS_REGION")
    
    log_success "Launched EC2 instance: $instance_id"
    
    # Wait for instance to be running
    log_info "Waiting for instance to be running..."
    aws ec2 wait instance-running \
        --instance-ids "$instance_id" \
        --region "$AWS_REGION"
    
    # Get public IP
    local public_ip=$(aws ec2 describe-instances \
        --instance-ids "$instance_id" \
        --query "Reservations[0].Instances[0].PublicIpAddress" \
        --output text \
        --region "$AWS_REGION")
    
    log_success "Instance is running at: $public_ip"
    
    echo "$instance_id:$public_ip"
}

setup_elastic_ip() {
    local instance_id=$1
    
    log_info "Allocating Elastic IP..."
    
    # Allocate EIP
    local allocation_id=$(aws ec2 allocate-address \
        --domain vpc \
        --query "AllocationId" \
        --output text \
        --region "$AWS_REGION")
    
    # Get EIP
    local eip=$(aws ec2 describe-addresses \
        --allocation-ids "$allocation_id" \
        --query "Addresses[0].PublicIp" \
        --output text \
        --region "$AWS_REGION")
    
    # Associate with instance
    aws ec2 associate-address \
        --instance-id "$instance_id" \
        --allocation-id "$allocation_id" \
        --region "$AWS_REGION"
    
    log_success "Associated Elastic IP: $eip"
    echo "$eip"
}

setup_route53() {
    local public_ip=$1
    local domain=$2
    
    log_info "Setting up Route53 DNS record..."
    
    # Get hosted zone ID
    local hosted_zone_id=$(aws route53 list-hosted-zones \
        --query "HostedZones[?Name=='${domain}.'].Id" \
        --output text \
        --region "$AWS_REGION" | sed 's|/hostedzone/||')
    
    if [ -z "$hosted_zone_id" ] || [ "$hosted_zone_id" == "None" ]; then
        log_warning "No hosted zone found for $domain. Skipping DNS setup."
        return 0
    fi
    
    # Create DNS record
    local change_batch=$(cat << EOF
{
    "Changes": [
        {
            "Action": "UPSERT",
            "ResourceRecordSet": {
                "Name": "bootstrap.${domain}",
                "Type": "A",
                "TTL": 300,
                "ResourceRecords": [{"Value": "${public_ip}"}]
            }
        }
    ]
}
EOF
)
    
    aws route53 change-resource-record-sets \
        --hosted-zone-id "$hosted_zone_id" \
        --change-batch "$change_batch" \
        --region "$AWS_REGION"
    
    log_success "Created DNS record: bootstrap.${domain} -> $public_ip"
}

setup_ssl_certificates() {
    local domain=$1
    
    log_info "Setting up SSL certificates..."
    
    # Check if certificate already exists
    local cert_arn=$(aws acm list-certificates \
        --query "CertificateSummaryList[?DomainName=='*.${domain}'].CertificateArn" \
        --output text \
        --region "$AWS_REGION")
    
    if [ -n "$cert_arn" ] && [ "$cert_arn" != "None" ]; then
        log_warning "SSL certificate already exists: $cert_arn"
        return 0
    fi
    
    # Request certificate
    cert_arn=$(aws acm request-certificate \
        --domain-name "*.${domain}" \
        --subject-alternative-names "${domain}" \
        --validation-method DNS \
        --query "CertificateArn" \
        --output text \
        --region "$AWS_REGION")
    
    log_success "Requested SSL certificate: $cert_arn"
    log_warning "Please validate the certificate via DNS validation"
}

# =============================================================================
# Main Deployment
# =============================================================================

deploy_bootstrap_server() {
    log_info "Starting PRSM Bootstrap Server deployment to AWS..."
    log_info "Region: $AWS_REGION"
    log_info "Environment: $ENVIRONMENT"
    log_info "Domain: $PRSM_DOMAIN"
    
    # Create network infrastructure
    local vpc_id=$(create_vpc)
    local subnet_id=$(create_subnet "$vpc_id")
    local igw_id=$(create_internet_gateway "$vpc_id")
    local rt_id=$(create_route_table "$vpc_id" "$igw_id" "$subnet_id")
    local sg_id=$(create_security_group "$vpc_id")
    
    # Launch instance
    local instance_info=$(launch_ec2_instance "$subnet_id" "$sg_id")
    local instance_id=$(echo "$instance_info" | cut -d: -f1)
    local public_ip=$(echo "$instance_info" | cut -d: -f2)
    
    # Setup Elastic IP
    local eip=$(setup_elastic_ip "$instance_id")
    
    # Setup DNS
    setup_route53 "$eip" "$PRSM_DOMAIN"
    
    # Setup SSL
    setup_ssl_certificates "$PRSM_DOMAIN"
    
    # Print summary
    echo ""
    log_success "=========================================="
    log_success "Deployment Complete!"
    log_success "=========================================="
    echo ""
    echo "Instance ID: $instance_id"
    echo "Public IP: $eip"
    echo "WebSocket URL: wss://bootstrap.${PRSM_DOMAIN}:8765"
    echo "API URL: https://bootstrap.${PRSM_DOMAIN}:8000"
    echo ""
    echo "Next Steps:"
    echo "1. Validate SSL certificate via DNS"
    echo "2. SSH to instance: ssh -i ~/.ssh/${KEY_NAME}.pem ec2-user@${eip}"
    echo "3. Check logs: docker logs prsm-bootstrap"
    echo "4. Configure monitoring: https://${eip}:3000"
    echo ""
}

# =============================================================================
# Argument Parsing
# =============================================================================

parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -r|--region)
                AWS_REGION="$2"
                shift 2
                ;;
            -e|--environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -d|--domain)
                PRSM_DOMAIN="$2"
                shift 2
                ;;
            -i|--instance-type)
                INSTANCE_TYPE="$2"
                shift 2
                ;;
            -k|--key-name)
                KEY_NAME="$2"
                shift 2
                ;;
            -h|--help)
                show_help
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                ;;
        esac
    done
}

# =============================================================================
# Script Entry Point
# =============================================================================

main() {
    # Set defaults from environment or defaults
    AWS_REGION="${AWS_REGION:-$DEFAULT_REGION}"
    ENVIRONMENT="${PRSM_ENV:-$DEFAULT_ENVIRONMENT}"
    PRSM_DOMAIN="${PRSM_DOMAIN:-$DEFAULT_DOMAIN}"
    INSTANCE_TYPE="${INSTANCE_TYPE:-$DEFAULT_INSTANCE_TYPE}"
    KEY_NAME="${KEY_NAME:-$DEFAULT_KEY_NAME}"
    
    # Parse command line arguments
    parse_arguments "$@"
    
    # Check prerequisites
    check_prerequisites
    
    # Check required environment variables
    if [ -z "${POSTGRES_PASSWORD:-}" ]; then
        log_warning "POSTGRES_PASSWORD not set, using default (not recommended for production)"
        POSTGRES_PASSWORD="prsm_secure_pass_$(date +%s)"
    fi
    
    if [ -z "${GRAFANA_ADMIN_PASSWORD:-}" ]; then
        log_warning "GRAFANA_ADMIN_PASSWORD not set, using default"
        GRAFANA_ADMIN_PASSWORD="admin"
    fi
    
    if [ -z "${PRSM_AUTH_SECRET:-}" ]; then
        PRSM_AUTH_SECRET=$(openssl rand -hex 32 2>/dev/null || uuidgen | tr -d '-')
        log_info "Generated PRSM_AUTH_SECRET"
    fi
    
    # Run deployment
    deploy_bootstrap_server
}

# Run main function
main "$@"
