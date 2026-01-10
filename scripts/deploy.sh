#!/bin/bash
#
# HelixForge Deployment Script
# Automates deployment to different environments
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_msg() { echo -e "${2}${1}${NC}"; }
print_info() { print_msg "$1" "$BLUE"; }
print_success() { print_msg "$1" "$GREEN"; }
print_warning() { print_msg "$1" "$YELLOW"; }
print_error() { print_msg "$1" "$RED"; }

# Configuration
APP_NAME="helixforge"
VERSION=$(grep "version" pyproject.toml | head -1 | cut -d'"' -f2)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Environment defaults
ENV="${HELIXFORGE_ENV:-development}"
DOCKER_REGISTRY="${DOCKER_REGISTRY:-}"
DEPLOY_DIR="${DEPLOY_DIR:-/opt/helixforge}"

# Print banner
print_banner() {
    echo ""
    echo "=================================="
    echo "  HelixForge Deployment v$VERSION"
    echo "=================================="
    echo ""
}

# Pre-deployment checks
pre_deploy_checks() {
    print_info "Running pre-deployment checks..."

    # Check if we're in the right directory
    if [ ! -f "pyproject.toml" ]; then
        print_error "Must run from HelixForge root directory"
        exit 1
    fi

    # Check environment file
    if [ "$ENV" != "development" ] && [ ! -f ".env" ]; then
        print_error ".env file required for $ENV deployment"
        exit 1
    fi

    # Run linting
    print_info "Running linter..."
    ruff check agents/ api/ models/ utils/ || {
        print_error "Linting failed. Fix issues before deploying."
        exit 1
    }

    # Run type checking
    print_info "Running type checker..."
    mypy agents/ api/ models/ utils/ --config-file=pyproject.toml || {
        print_warning "Type check warnings found (non-blocking)"
    }

    # Run tests
    print_info "Running tests..."
    pytest tests/ -v --tb=short -x || {
        print_error "Tests failed. Fix before deploying."
        exit 1
    }

    print_success "Pre-deployment checks passed"
}

# Build Docker image
build_docker() {
    print_info "Building Docker image..."

    IMAGE_TAG="${APP_NAME}:${VERSION}"
    IMAGE_TAG_LATEST="${APP_NAME}:latest"

    docker build -t "$IMAGE_TAG" -t "$IMAGE_TAG_LATEST" .

    print_success "Docker image built: $IMAGE_TAG"

    # Push to registry if configured
    if [ -n "$DOCKER_REGISTRY" ]; then
        print_info "Pushing to registry: $DOCKER_REGISTRY"
        docker tag "$IMAGE_TAG" "$DOCKER_REGISTRY/$IMAGE_TAG"
        docker tag "$IMAGE_TAG_LATEST" "$DOCKER_REGISTRY/$IMAGE_TAG_LATEST"
        docker push "$DOCKER_REGISTRY/$IMAGE_TAG"
        docker push "$DOCKER_REGISTRY/$IMAGE_TAG_LATEST"
        print_success "Image pushed to registry"
    fi
}

# Deploy with Docker Compose
deploy_docker_compose() {
    print_info "Deploying with Docker Compose..."

    # Select compose file based on environment
    COMPOSE_FILE="docker-compose.yaml"
    if [ -f "docker-compose.${ENV}.yaml" ]; then
        COMPOSE_FILE="docker-compose.${ENV}.yaml"
    fi

    # Stop existing containers
    docker compose -f "$COMPOSE_FILE" down || true

    # Pull latest images
    docker compose -f "$COMPOSE_FILE" pull

    # Start services
    docker compose -f "$COMPOSE_FILE" up -d

    # Wait for services to be healthy
    print_info "Waiting for services to start..."
    sleep 10

    # Health check
    if curl -s http://localhost:8000/health | grep -q "healthy"; then
        print_success "Deployment successful! API is healthy."
    else
        print_error "Health check failed"
        docker compose -f "$COMPOSE_FILE" logs --tail=50
        exit 1
    fi
}

# Deploy to bare metal / VM
deploy_direct() {
    print_info "Deploying directly to $DEPLOY_DIR..."

    # Create deployment directory
    sudo mkdir -p "$DEPLOY_DIR"

    # Copy files
    sudo rsync -av --exclude='.git' --exclude='venv' --exclude='__pycache__' \
        --exclude='.pytest_cache' --exclude='*.pyc' \
        ./ "$DEPLOY_DIR/"

    # Set permissions
    sudo chown -R www-data:www-data "$DEPLOY_DIR" 2>/dev/null || true

    # Install dependencies
    cd "$DEPLOY_DIR"
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt

    # Restart service (systemd)
    if systemctl is-active --quiet helixforge; then
        sudo systemctl restart helixforge
        print_success "Service restarted"
    else
        print_warning "Systemd service not configured. Start manually with:"
        echo "  cd $DEPLOY_DIR && source venv/bin/activate && uvicorn api.server:app --host 0.0.0.0 --port 8000"
    fi
}

# Create backup before deployment
create_backup() {
    print_info "Creating backup..."

    BACKUP_DIR="backups/${TIMESTAMP}"
    mkdir -p "$BACKUP_DIR"

    # Backup database (if configured)
    if [ -n "$DB_HOST" ]; then
        pg_dump -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" > "$BACKUP_DIR/database.sql" || true
    fi

    # Backup configuration
    cp .env "$BACKUP_DIR/.env" 2>/dev/null || true
    cp config.yaml "$BACKUP_DIR/config.yaml" 2>/dev/null || true

    print_success "Backup created: $BACKUP_DIR"
}

# Rollback to previous version
rollback() {
    print_info "Rolling back..."

    # Find latest backup
    LATEST_BACKUP=$(ls -td backups/*/ 2>/dev/null | head -1)

    if [ -z "$LATEST_BACKUP" ]; then
        print_error "No backup found for rollback"
        exit 1
    fi

    print_info "Rolling back to: $LATEST_BACKUP"

    # Restore database
    if [ -f "$LATEST_BACKUP/database.sql" ] && [ -n "$DB_HOST" ]; then
        psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" < "$LATEST_BACKUP/database.sql"
    fi

    # Restore configuration
    cp "$LATEST_BACKUP/.env" .env 2>/dev/null || true
    cp "$LATEST_BACKUP/config.yaml" config.yaml 2>/dev/null || true

    # Restart services
    docker compose down && docker compose up -d

    print_success "Rollback complete"
}

# Print deployment info
print_info_summary() {
    echo ""
    echo "=================================="
    echo "  Deployment Summary"
    echo "=================================="
    echo "  Environment: $ENV"
    echo "  Version:     $VERSION"
    echo "  Timestamp:   $TIMESTAMP"
    echo "=================================="
    echo ""
    echo "  API URL:     http://localhost:8000"
    echo "  Docs:        http://localhost:8000/docs"
    echo "  Health:      http://localhost:8000/health"
    echo "  Metrics:     http://localhost:8000/metrics"
    echo ""
}

# Show usage
usage() {
    echo "Usage: ./deploy.sh [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  docker      Deploy using Docker Compose (default)"
    echo "  direct      Deploy directly to server"
    echo "  build       Build Docker image only"
    echo "  rollback    Rollback to previous version"
    echo "  check       Run pre-deployment checks only"
    echo ""
    echo "Options:"
    echo "  --env ENV   Set environment (development/staging/production)"
    echo "  --skip-tests  Skip running tests"
    echo "  --skip-backup Skip creating backup"
    echo ""
    echo "Examples:"
    echo "  ./deploy.sh docker --env production"
    echo "  ./deploy.sh direct --env staging"
    echo "  ./deploy.sh rollback"
}

# Main
main() {
    print_banner

    COMMAND="${1:-docker}"
    SKIP_TESTS=false
    SKIP_BACKUP=false

    # Parse options
    shift || true
    while [[ $# -gt 0 ]]; do
        case $1 in
            --env)
                ENV="$2"
                shift 2
                ;;
            --skip-tests)
                SKIP_TESTS=true
                shift
                ;;
            --skip-backup)
                SKIP_BACKUP=true
                shift
                ;;
            --help|-h)
                usage
                exit 0
                ;;
            *)
                shift
                ;;
        esac
    done

    export HELIXFORGE_ENV="$ENV"

    case $COMMAND in
        docker)
            [ "$SKIP_TESTS" = false ] && pre_deploy_checks
            [ "$SKIP_BACKUP" = false ] && create_backup
            build_docker
            deploy_docker_compose
            print_info_summary
            ;;
        direct)
            [ "$SKIP_TESTS" = false ] && pre_deploy_checks
            [ "$SKIP_BACKUP" = false ] && create_backup
            deploy_direct
            print_info_summary
            ;;
        build)
            build_docker
            ;;
        rollback)
            rollback
            ;;
        check)
            pre_deploy_checks
            ;;
        *)
            print_error "Unknown command: $COMMAND"
            usage
            exit 1
            ;;
    esac
}

main "$@"
