#!/bin/bash
#
# HelixForge Setup Script
# Cross-Dataset Insight Synthesizer
#
# This script sets up HelixForge for development or production use.
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored message
print_msg() {
    echo -e "${2}${1}${NC}"
}

print_info() {
    print_msg "$1" "$BLUE"
}

print_success() {
    print_msg "$1" "$GREEN"
}

print_warning() {
    print_msg "$1" "$YELLOW"
}

print_error() {
    print_msg "$1" "$RED"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Print banner
print_banner() {
    echo ""
    echo "  _   _      _ _      ______                    "
    echo " | | | |    | (_)     |  ___|                   "
    echo " | |_| | ___| |___  __| |_ ___  _ __ __ _  ___  "
    echo " |  _  |/ _ \ | \ \/ /|  _/ _ \| '__/ _\` |/ _ \ "
    echo " | | | |  __/ | |>  < | || (_) | | | (_| |  __/ "
    echo " \_| |_/\___|_|_/_/\_\\_| \___/|_|  \__, |\___| "
    echo "                                     __/ |      "
    echo "                                    |___/       "
    echo ""
    echo "  Cross-Dataset Insight Synthesizer v1.0.0"
    echo ""
}

# Check prerequisites
check_prerequisites() {
    print_info "Checking prerequisites..."

    # Check Python
    if ! command_exists python3; then
        print_error "Python 3 is not installed. Please install Python 3.10 or higher."
        exit 1
    fi

    PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    if [[ $(echo "$PYTHON_VERSION < 3.10" | bc -l 2>/dev/null || echo "0") == "1" ]]; then
        print_warning "Python version $PYTHON_VERSION detected. Python 3.10+ recommended."
    else
        print_success "Python $PYTHON_VERSION found"
    fi

    # Check pip
    if ! command_exists pip3; then
        print_error "pip3 is not installed."
        exit 1
    fi
    print_success "pip3 found"

    # Check Docker (optional)
    if command_exists docker; then
        print_success "Docker found (optional)"
    else
        print_warning "Docker not found. Docker is optional but recommended for full deployment."
    fi

    # Check Docker Compose (optional)
    if command_exists docker-compose || docker compose version >/dev/null 2>&1; then
        print_success "Docker Compose found (optional)"
    else
        print_warning "Docker Compose not found. Optional for full deployment."
    fi

    echo ""
}

# Create virtual environment
create_venv() {
    print_info "Creating virtual environment..."

    if [ -d "venv" ]; then
        print_warning "Virtual environment already exists. Skipping creation."
    else
        python3 -m venv venv
        print_success "Virtual environment created"
    fi

    # Activate virtual environment
    source venv/bin/activate
    print_success "Virtual environment activated"
}

# Install dependencies
install_dependencies() {
    print_info "Installing dependencies..."

    # Upgrade pip
    pip install --upgrade pip

    # Install requirements
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
        print_success "Dependencies installed"
    else
        print_error "requirements.txt not found"
        exit 1
    fi

    # Install development dependencies
    if [ "$1" == "dev" ]; then
        print_info "Installing development dependencies..."
        pip install pytest pytest-cov pytest-asyncio ruff mypy types-requests types-PyYAML
        print_success "Development dependencies installed"
    fi
}

# Create directories
create_directories() {
    print_info "Creating directories..."

    mkdir -p data/raw
    mkdir -p data/processed
    mkdir -p outputs/reports
    mkdir -p outputs/visualizations
    mkdir -p logs

    print_success "Directories created"
}

# Create environment file
create_env_file() {
    print_info "Setting up environment..."

    if [ -f ".env" ]; then
        print_warning ".env file already exists. Skipping."
    else
        cat > .env << 'EOF'
# HelixForge Environment Configuration
# Copy this file to .env and fill in your values

# OpenAI API Key (required)
OPENAI_API_KEY=your-openai-api-key-here

# Environment
HELIXFORGE_ENV=development

# Database (optional - uses SQLite by default)
# DB_HOST=localhost
# DB_PORT=5432
# DB_NAME=helixforge
# DB_USER=helixforge
# DB_PASSWORD=your-password

# Neo4j Graph Database (optional)
# GRAPH_URI=bolt://localhost:7687
# GRAPH_USER=neo4j
# GRAPH_PASSWORD=your-password

# Weaviate Vector Store (optional)
# WEAVIATE_HOST=localhost
# WEAVIATE_PORT=8080

# Sentry Error Tracking (optional)
# SENTRY_DSN=your-sentry-dsn

# API Settings
# API_KEY=your-api-key-for-authentication
EOF
        print_success ".env file created. Please edit it with your API keys."
    fi
}

# Verify installation
verify_installation() {
    print_info "Verifying installation..."

    # Try importing main modules
    python3 -c "from agents.data_ingestor_agent import DataIngestorAgent; print('Agents: OK')" || {
        print_error "Failed to import agents"
        exit 1
    }

    python3 -c "from api.server import app; print('API: OK')" || {
        print_error "Failed to import API"
        exit 1
    }

    python3 -c "from models.schemas import IngestResult; print('Models: OK')" || {
        print_error "Failed to import models"
        exit 1
    }

    print_success "Installation verified"
}

# Run tests
run_tests() {
    print_info "Running tests..."
    pytest tests/ -v --tb=short -x || {
        print_warning "Some tests failed. Check the output above."
    }
}

# Print usage instructions
print_usage() {
    echo ""
    print_success "HelixForge setup complete!"
    echo ""
    echo "Next steps:"
    echo ""
    echo "  1. Edit .env file with your OpenAI API key:"
    echo "     \$ nano .env"
    echo ""
    echo "  2. Activate the virtual environment:"
    echo "     \$ source venv/bin/activate"
    echo ""
    echo "  3. Start the API server:"
    echo "     \$ make run"
    echo "     or"
    echo "     \$ uvicorn api.server:app --reload"
    echo ""
    echo "  4. Open the API documentation:"
    echo "     http://localhost:8000/docs"
    echo ""
    echo "  For Docker deployment:"
    echo "     \$ docker compose up -d"
    echo ""
    echo "  For more information, see README.md"
    echo ""
}

# Main setup function
main() {
    print_banner

    # Parse arguments
    MODE="prod"
    SKIP_TESTS=false

    while [[ $# -gt 0 ]]; do
        case $1 in
            --dev)
                MODE="dev"
                shift
                ;;
            --skip-tests)
                SKIP_TESTS=true
                shift
                ;;
            --help|-h)
                echo "Usage: ./setup.sh [OPTIONS]"
                echo ""
                echo "Options:"
                echo "  --dev         Install development dependencies"
                echo "  --skip-tests  Skip running tests after installation"
                echo "  --help, -h    Show this help message"
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done

    # Run setup steps
    check_prerequisites
    create_venv
    install_dependencies "$MODE"
    create_directories
    create_env_file
    verify_installation

    if [ "$SKIP_TESTS" = false ] && [ "$MODE" = "dev" ]; then
        run_tests
    fi

    print_usage
}

# Run main
main "$@"
