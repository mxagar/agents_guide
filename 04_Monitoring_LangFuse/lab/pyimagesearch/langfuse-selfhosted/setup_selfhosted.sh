#!/bin/bash

# =============================================================================
# Automated Setup Script for Self-Hosted Langfuse
# =============================================================================

set -e  # Exit on error

echo "======================================================================"
echo "Self-Hosted Langfuse Setup"
echo "======================================================================"
echo ""

# Check prerequisites
echo "📋 Checking prerequisites..."

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker first."
    exit 1
fi
echo "✅ Docker installed"

# Check Docker Compose
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose not found. Please install Docker Compose first."
    exit 1
fi
echo "✅ Docker Compose installed"

# Check Python
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo "❌ Python not found. Please install Python 3.8+ first."
    exit 1
fi
echo "✅ Python installed"

echo ""

# =============================================================================
# Step 1: Create .env file
# =============================================================================
echo "1️⃣  Setting up environment configuration..."

if [ ! -f .env ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "✅ Created .env file"
    echo ""
    echo "⚠️  IMPORTANT: You need to configure API keys after setup!"
    echo "   1. Start services with: docker-compose up -d"
    echo "   2. Open http://localhost:3000"
    echo "   3. Create account and project"
    echo "   4. Get API keys from Settings → API Keys"
    echo "   5. Edit .env and add your keys"
else
    echo "✅ .env file already exists"
fi

echo ""

# =============================================================================
# Step 2: Start Docker services
# =============================================================================
echo "2️⃣  Starting Docker services..."
echo "   This may take a few minutes on first run..."
echo ""

# Check if GPU is available
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected, using GPU profile"
    docker-compose --profile gpu up -d
else
    echo "⚠️  No GPU detected, using CPU profile (slower)"
    docker-compose --profile cpu up -d
fi

echo ""
echo "✅ Docker services started"
echo ""

# =============================================================================
# Step 3: Wait for services to be healthy
# =============================================================================
echo "3️⃣  Waiting for services to become healthy..."
echo "   This can take 30-60 seconds on first start..."
echo ""

MAX_WAIT=120
ELAPSED=0

while [ $ELAPSED -lt $MAX_WAIT ]; do
    if curl -f -s http://localhost:3000/api/public/health > /dev/null 2>&1; then
        echo "✅ Langfuse server is healthy!"
        break
    fi
    
    echo -n "."
    sleep 5
    ELAPSED=$((ELAPSED + 5))
done

if [ $ELAPSED -ge $MAX_WAIT ]; then
    echo ""
    echo "⚠️  Langfuse took longer than expected to start"
    echo "   Check logs: docker-compose logs langfuse-server"
    echo ""
fi

echo ""

# =============================================================================
# Step 4: Check vLLM
# =============================================================================
echo "4️⃣  Checking vLLM server..."

if curl -f -s http://localhost:8000/v1/models > /dev/null 2>&1; then
    echo "✅ vLLM server is running"
else
    echo "⚠️  vLLM server not responding yet"
    echo "   This is normal - vLLM may take several minutes to load the model"
    echo "   Check status: docker-compose logs -f vllm"
fi

echo ""

# =============================================================================
# Step 5: Python environment
# =============================================================================
echo "5️⃣  Setting up Python environment..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    if command -v python3 &> /dev/null; then
        python3 -m venv venv
    else
        python -m venv venv
    fi
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate and install dependencies
echo "Installing Python dependencies..."
source venv/bin/activate 2>/dev/null || . venv/Scripts/activate 2>/dev/null

pip install --upgrade pip > /dev/null 2>&1
pip install -r requirements.txt > /dev/null 2>&1

echo "✅ Python dependencies installed"
echo ""

# =============================================================================
# Summary
# =============================================================================
echo "======================================================================"
echo "🎉 SETUP COMPLETE!"
echo "======================================================================"
echo ""
echo "📊 Services Status:"
docker-compose ps
echo ""
echo "🔧 Next Steps:"
echo ""
echo "1. Open Langfuse UI:"
echo "   👉 http://localhost:3000"
echo ""
echo "2. Create your account and project"
echo ""
echo "3. Get API keys:"
echo "   - Go to Settings → API Keys"
echo "   - Copy Public Key and Secret Key"
echo ""
echo "4. Configure .env file:"
echo "   - Edit .env"
echo "   - Add your LANGFUSE_PUBLIC_KEY"
echo "   - Add your LANGFUSE_SECRET_KEY"
echo ""
echo "5. Activate Python environment:"
echo "   source venv/bin/activate"
echo ""
echo "6. Run health check:"
echo "   python check_selfhosted_health.py"
echo ""
echo "7. Test the setup:"
echo "   python diagnose_langfuse.py"
echo ""
echo "8. Try examples:"
echo "   python src/tracing_decorator.py"
echo ""
echo "======================================================================"
echo "📚 Documentation: README.md"
echo "🔍 Logs: docker-compose logs -f"
echo "🛑 Stop: docker-compose down"
echo "======================================================================"
