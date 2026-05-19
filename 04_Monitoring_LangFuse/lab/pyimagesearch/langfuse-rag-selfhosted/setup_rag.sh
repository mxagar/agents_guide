#!/bin/bash

# =============================================================================
# Self-Hosted RAG with Langfuse - Setup Script
# =============================================================================

set -e

echo ""
echo "======================================================================"
echo "  Self-Hosted RAG with Langfuse - Setup Script"
echo "======================================================================"
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Error: Docker is not running"
    echo "Please start Docker Desktop and try again."
    exit 1
fi

echo "✅ Docker is running"
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "📝 Creating .env file from template..."
    if [ -f .env.example ]; then
        cp .env.example .env
        echo "✅ Created .env file"
        echo ""
        echo "⚠️  IMPORTANT: You need to configure your Langfuse keys!"
        echo ""
        echo "Next steps:"
        echo "1. Start Langfuse: docker-compose up -d"
        echo "2. Open http://localhost:3000"
        echo "3. Create an account and project"
        echo "4. Copy your project keys"
        echo "5. Edit .env and add your keys"
        echo "6. Run this script again"
        echo ""
        exit 0
    else
        echo "❌ Error: .env.example not found"
        exit 1
    fi
else
    echo "✅ .env file exists"
fi

# Check if Langfuse keys are configured
LANGFUSE_PUBLIC_KEY=$(grep "^LANGFUSE_PUBLIC_KEY=" .env | cut -d'=' -f2 | tr -d '"' | tr -d "'")
LANGFUSE_SECRET_KEY=$(grep "^LANGFUSE_SECRET_KEY=" .env | cut -d'=' -f2 | tr -d '"' | tr -d "'")

if [ "$LANGFUSE_PUBLIC_KEY" = "your-public-key-here" ] || [ -z "$LANGFUSE_PUBLIC_KEY" ]; then
    echo "⚠️  Langfuse keys not configured in .env"
    echo ""
    echo "To configure:"
    echo "1. Start Langfuse: docker-compose up -d"
    echo "2. Open http://localhost:3000"
    echo "3. Create an account and project"
    echo "4. Copy your project keys"
    echo "5. Edit .env and add your keys"
    echo ""
    
    read -p "Do you want to continue without configuring keys? (y/N) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 0
    fi
else
    echo "✅ Langfuse keys configured"
fi

echo ""

# Check which Docker profile to use
echo "======================================================================"
echo "  Docker Services Setup"
echo "======================================================================"
echo ""
echo "Which vLLM profile do you want to use?"
echo "1) GPU (recommended for Linux with NVIDIA GPU)"
echo "2) CPU (fallback for Mac or systems without GPU)"
echo "3) Skip vLLM (only start Langfuse)"
echo ""
read -p "Enter choice (1-3): " choice

case $choice in
    1)
        PROFILE="gpu"
        echo ""
        echo "🚀 Starting Langfuse + vLLM (GPU mode)..."
        docker-compose --profile gpu up -d
        ;;
    2)
        PROFILE="cpu"
        echo ""
        echo "🚀 Starting Langfuse + vLLM (CPU mode)..."
        docker-compose --profile cpu up -d
        ;;
    3)
        PROFILE="none"
        echo ""
        echo "🚀 Starting Langfuse only..."
        docker-compose up -d
        ;;
    *)
        echo "❌ Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "⏳ Waiting for services to start..."
sleep 5

# Check service status
echo ""
echo "======================================================================"
echo "  Service Status"
echo "======================================================================"
echo ""
docker-compose ps

echo ""
echo "======================================================================"
echo "  Health Checks"
echo "======================================================================"
echo ""

# Check Langfuse
echo "Checking Langfuse..."
if curl -s -f http://localhost:3000/api/public/health > /dev/null 2>&1; then
    echo "✅ Langfuse is healthy at http://localhost:3000"
else
    echo "⚠️  Langfuse health check failed (may still be starting)"
    echo "   Check logs: docker-compose logs langfuse-server"
fi

# Check vLLM if started
if [ "$PROFILE" != "none" ]; then
    echo ""
    echo "Checking vLLM..."
    if curl -s -f http://localhost:8000/health > /dev/null 2>&1; then
        echo "✅ vLLM is healthy at http://localhost:8000"
    else
        echo "⚠️  vLLM health check failed (may still be starting)"
        if [ "$PROFILE" = "gpu" ]; then
            echo "   Check logs: docker-compose logs vllm-gpu"
        else
            echo "   Check logs: docker-compose logs vllm-cpu"
        fi
        echo "   Note: vLLM can take 2-3 minutes to download model and start"
    fi
fi

echo ""
echo "======================================================================"
echo "  Python Environment Setup"
echo "======================================================================"
echo ""

# Check if Python dependencies are installed
echo "Checking Python dependencies..."
if python3 -c "import langfuse" 2>/dev/null; then
    echo "✅ Python dependencies already installed"
else
    echo "📦 Installing Python dependencies..."
    pip install -r requirements.txt
    echo "✅ Python dependencies installed"
fi

echo ""
echo "======================================================================"
echo "  Setup Complete!"
echo "======================================================================"
echo ""
echo "✅ Services are starting up"
echo ""
echo "🌐 Access Points:"
echo "   Langfuse UI:  http://localhost:3000"
if [ "$PROFILE" != "none" ]; then
    echo "   vLLM API:     http://localhost:8000"
fi
echo ""
echo "📚 Next Steps:"
echo ""
echo "1. Configure Langfuse (if not done):"
echo "   - Open http://localhost:3000"
echo "   - Create account and project"
echo "   - Copy keys to .env file"
echo ""
echo "2. Test the setup:"
echo "   cd src"
echo "   python llm_client.py          # Test LLM client"
echo "   python retriever.py           # Test retrieval"
echo "   python rag_pipeline.py        # Test full RAG pipeline"
echo "   python agent_orchestration.py # Test agent workflow"
echo ""
echo "3. View traces in Langfuse dashboard"
echo ""
echo "🔧 Useful Commands:"
echo "   docker-compose ps                    # Check status"
echo "   docker-compose logs -f               # View logs"
echo "   docker-compose down                  # Stop services"
if [ "$PROFILE" != "none" ]; then
    echo "   curl http://localhost:8000/v1/models # List vLLM models"
fi
echo ""
echo "📖 For troubleshooting, see README.md"
echo ""
