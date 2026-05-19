"""
Health Check Script for Self-Hosted Langfuse Setup
Verifies all services are running correctly
"""

import os
import sys
import time
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("="*70)
print("SELF-HOSTED LANGFUSE HEALTH CHECK")
print("="*70)
print()

all_healthy = True

# ============================================================================
# 1. CHECK DOCKER SERVICES
# ============================================================================
print("1️⃣  CHECKING DOCKER SERVICES")
print("-" * 70)

try:
    import subprocess
    result = subprocess.run(
        ["docker-compose", "ps", "--format", "json"],
        cwd=os.path.dirname(os.path.abspath(__file__)),
        capture_output=True,
        text=True,
        timeout=10
    )
    
    if result.returncode == 0:
        print("✅ Docker Compose is running")
        # Parse and show service status
        import json
        try:
            services = [json.loads(line) for line in result.stdout.strip().split('\n') if line]
            for svc in services:
                name = svc.get('Service', svc.get('Name', 'unknown'))
                state = svc.get('State', 'unknown')
                status = svc.get('Status', '')
                
                if 'running' in state.lower() or 'up' in state.lower():
                    print(f"  ✅ {name}: {state}")
                else:
                    print(f"  ❌ {name}: {state} - {status}")
                    all_healthy = False
        except:
            print("  ℹ️  Services are running (unable to parse details)")
    else:
        print("❌ Docker Compose not running or error checking status")
        print("   Run: docker-compose up -d")
        all_healthy = False
except FileNotFoundError:
    print("❌ docker-compose command not found")
    print("   Install Docker Compose: https://docs.docker.com/compose/install/")
    all_healthy = False
except Exception as e:
    print(f"⚠️  Could not check Docker status: {e}")

print()

# ============================================================================
# 2. CHECK LANGFUSE SERVER
# ============================================================================
print("2️⃣  CHECKING LANGFUSE SERVER")
print("-" * 70)

langfuse_host = os.getenv("LANGFUSE_HOST", "http://localhost:3000")

try:
    response = requests.get(f"{langfuse_host}/api/public/health", timeout=10)
    
    if response.status_code == 200:
        print(f"✅ Langfuse server is healthy at {langfuse_host}")
        print(f"   UI available at: {langfuse_host}")
    else:
        print(f"⚠️  Unexpected response from Langfuse: {response.status_code}")
        all_healthy = False
        
except requests.exceptions.ConnectionError:
    print(f"❌ Cannot connect to Langfuse at {langfuse_host}")
    print("   Make sure Docker services are running: docker-compose up -d")
    print("   Wait 30-60 seconds for initialization on first start")
    all_healthy = False
except Exception as e:
    print(f"❌ Error checking Langfuse: {e}")
    all_healthy = False

print()

# ============================================================================
# 3. CHECK POSTGRESQL
# ============================================================================
print("3️⃣  CHECKING POSTGRESQL DATABASE")
print("-" * 70)

try:
    result = subprocess.run(
        ["docker", "exec", "langfuse-postgres", "pg_isready", "-U", "langfuse"],
        capture_output=True,
        text=True,
        timeout=10
    )
    
    if result.returncode == 0:
        print("✅ PostgreSQL is accepting connections")
    else:
        print("❌ PostgreSQL is not ready")
        print(f"   Output: {result.stdout}")
        all_healthy = False
except Exception as e:
    print(f"⚠️  Could not check PostgreSQL: {e}")
    all_healthy = False

print()

# ============================================================================
# 4. CHECK vLLM SERVER
# ============================================================================
print("4️⃣  CHECKING vLLM SERVER")
print("-" * 70)

vllm_url = os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1")

try:
    response = requests.get(f"{vllm_url}/models", timeout=10)
    
    if response.status_code == 200:
        models = response.json()
        model_list = models.get('data', [])
        print(f"✅ vLLM server is running at {vllm_url}")
        print(f"   Available models: {len(model_list)}")
        for model in model_list:
            print(f"     - {model.get('id', 'unknown')}")
    else:
        print(f"⚠️  Unexpected response from vLLM: {response.status_code}")
        all_healthy = False
        
except requests.exceptions.ConnectionError:
    print(f"❌ Cannot connect to vLLM at {vllm_url}")
    print("   Make sure vLLM is running: docker-compose up -d vllm")
    print("   Check logs: docker-compose logs vllm")
    all_healthy = False
except Exception as e:
    print(f"❌ Error checking vLLM: {e}")
    all_healthy = False

print()

# ============================================================================
# 5. CHECK ENVIRONMENT VARIABLES
# ============================================================================
print("5️⃣  CHECKING ENVIRONMENT CONFIGURATION")
print("-" * 70)

required_vars = {
    "LANGFUSE_PUBLIC_KEY": os.getenv("LANGFUSE_PUBLIC_KEY"),
    "LANGFUSE_SECRET_KEY": os.getenv("LANGFUSE_SECRET_KEY"),
    "LANGFUSE_HOST": os.getenv("LANGFUSE_HOST"),
    "OPENAI_BASE_URL": os.getenv("OPENAI_BASE_URL"),
}

for var_name, var_value in required_vars.items():
    if var_value:
        display_value = var_value[:30] + "..." if len(var_value) > 30 else var_value
        print(f"✅ {var_name}: {display_value}")
    else:
        print(f"❌ {var_name}: NOT SET")
        all_healthy = False

if not os.getenv("LANGFUSE_PUBLIC_KEY") or not os.getenv("LANGFUSE_SECRET_KEY"):
    print()
    print("⚠️  API keys not configured!")
    print("   1. Open http://localhost:3000")
    print("   2. Create account and project")
    print("   3. Get API keys from Settings → API Keys")
    print("   4. Add them to .env file")

print()

# ============================================================================
# 6. VALIDATE API CREDENTIALS (if set)
# ============================================================================
if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
    print("6️⃣  VALIDATING API CREDENTIALS")
    print("-" * 70)
    
    try:
        response = requests.get(
            f"{langfuse_host}/api/public/projects",
            auth=(os.getenv("LANGFUSE_PUBLIC_KEY"), os.getenv("LANGFUSE_SECRET_KEY")),
            timeout=10
        )
        
        if response.status_code == 200:
            projects = response.json()
            print(f"✅ API credentials are valid")
            print(f"✅ Found {len(projects.get('data', []))} project(s):")
            for proj in projects.get('data', []):
                print(f"   - {proj.get('name')} (ID: {proj.get('id')})")
        elif response.status_code == 401:
            print("❌ Invalid API credentials!")
            print("   Check that PUBLIC_KEY and SECRET_KEY match your Langfuse project")
            all_healthy = False
        else:
            print(f"⚠️  Unexpected response: {response.status_code}")
            all_healthy = False
    except Exception as e:
        print(f"⚠️  Could not validate credentials: {e}")
        all_healthy = False
    
    print()

# ============================================================================
# SUMMARY
# ============================================================================
print("="*70)
if all_healthy:
    print("🎉 ALL SYSTEMS HEALTHY!")
    print()
    print("Next steps:")
    print("  1. Run diagnostic test: python diagnose_langfuse.py")
    print("  2. Try examples: python src/tracing_decorator.py")
    print(f"  3. View traces: {langfuse_host}")
else:
    print("⚠️  SOME ISSUES DETECTED")
    print()
    print("Troubleshooting:")
    print("  1. Start services: docker-compose up -d")
    print("  2. Check logs: docker-compose logs -f")
    print("  3. Wait 30-60 seconds for initialization")
    print("  4. Configure API keys in .env file")
    print("  5. Re-run: python check_selfhosted_health.py")

print("="*70)

sys.exit(0 if all_healthy else 1)
