#!/usr/bin/env python3
"""
Health check for Langfuse RAG Self-Hosted Setup

Verifies all components are working correctly.
"""

import os
import sys
from pathlib import Path

# Load environment variables
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✅ Loaded .env from {env_path}\n")
    else:
        print(f"⚠️  No .env file found at {env_path}\n")
except ImportError:
    print("⚠️  python-dotenv not installed\n")

print("=" * 70)
print("  Langfuse RAG Self-Hosted Health Check")
print("=" * 70)
print()

# Check 1: Environment Variables
print("1️⃣  Checking Environment Variables...")
print("-" * 70)

required_vars = {
    "LANGFUSE_HOST": "http://localhost:3000",
    "LANGFUSE_PUBLIC_KEY": None,
    "LANGFUSE_SECRET_KEY": None,
    "OPENAI_BASE_URL": "http://localhost:8000/v1",
    "OPENAI_API_KEY": "dummy"
}

all_vars_ok = True
for var, default in required_vars.items():
    value = os.getenv(var, default)
    if value:
        # Mask secrets
        if "KEY" in var or "SECRET" in var:
            display = f"{value[:20]}..." if len(value) > 20 else value
        else:
            display = value
        print(f"  ✅ {var}: {display}")
    else:
        print(f"  ❌ {var}: NOT SET")
        all_vars_ok = False

print()

# Check 2: Docker Services
print("2️⃣  Checking Docker Services...")
print("-" * 70)

import subprocess

try:
    result = subprocess.run(
        ["docker-compose", "ps", "--format", "json"],
        cwd=Path(__file__).parent,
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print("  ✅ Docker Compose is available")
        
        # Parse running services
        import json
        services = []
        for line in result.stdout.strip().split('\n'):
            if line:
                try:
                    service = json.loads(line)
                    services.append(service)
                except:
                    pass
        
        if services:
            print(f"  ✅ Found {len(services)} running service(s):")
            for svc in services:
                name = svc.get('Service', svc.get('Name', 'unknown'))
                state = svc.get('State', 'unknown')
                health = svc.get('Health', 'N/A')
                print(f"     - {name}: {state} (health: {health})")
        else:
            print("  ⚠️  No services running")
            print("     Run: docker-compose --profile gpu up -d")
    else:
        print("  ⚠️  Could not check docker services")
except Exception as e:
    print(f"  ⚠️  Error checking Docker: {e}")

print()

# Check 3: Service Endpoints
print("3️⃣  Checking Service Endpoints...")
print("-" * 70)

import urllib.request
import urllib.error

endpoints = [
    ("Langfuse", "http://localhost:3000/api/public/health"),
    ("vLLM", "http://localhost:8000/health"),
]

for name, url in endpoints:
    try:
        req = urllib.request.Request(url, method='GET')
        with urllib.request.urlopen(req, timeout=5) as response:
            status = response.status
            if status == 200:
                print(f"  ✅ {name} is healthy: {url}")
            else:
                print(f"  ⚠️  {name} returned status {status}: {url}")
    except urllib.error.URLError as e:
        print(f"  ❌ {name} is not accessible: {url}")
        print(f"     Error: {e.reason}")
    except Exception as e:
        print(f"  ❌ {name} check failed: {e}")

print()

# Check 4: Python Dependencies
print("4️⃣  Checking Python Dependencies...")
print("-" * 70)

required_packages = [
    "langfuse",
    "openai",
    "dotenv",
    "yaml",
    "sentence_transformers",
    "faiss",
    "numpy"
]

all_deps_ok = True
for package in required_packages:
    try:
        if package == "dotenv":
            __import__("dotenv")
        elif package == "yaml":
            __import__("yaml")
        else:
            __import__(package.replace("-", "_"))
        print(f"  ✅ {package}")
    except ImportError:
        print(f"  ❌ {package} - NOT INSTALLED")
        all_deps_ok = False

if not all_deps_ok:
    print("\n  Install missing packages:")
    print("    pip install -r requirements.txt")

print()

# Check 5: Project Structure
print("5️⃣  Checking Project Structure...")
print("-" * 70)

required_files = [
    "configs/config.yaml",
    "data/sample_docs.txt",
    "src/llm_client.py",
    "src/retriever.py",
    "src/rag_pipeline.py",
    "src/agent_orchestration.py",
    "src/evaluation.py",
    "src/config.py",
    "src/llm_utils.py",
    "src/langfuse_instrumentation.py",
]

project_root = Path(__file__).parent
all_files_ok = True
for file_path in required_files:
    full_path = project_root / file_path
    if full_path.exists():
        print(f"  ✅ {file_path}")
    else:
        print(f"  ❌ {file_path} - NOT FOUND")
        all_files_ok = False

print()

# Summary
print("=" * 70)
print("  Summary")
print("=" * 70)

issues = []
if not all_vars_ok:
    issues.append("Environment variables not configured")
if not all_deps_ok:
    issues.append("Python dependencies missing")
if not all_files_ok:
    issues.append("Project files missing")

if issues:
    print("\n⚠️  Issues found:")
    for issue in issues:
        print(f"  - {issue}")
    print("\n📖 See README.md for setup instructions")
    sys.exit(1)
else:
    print("\n✅ All checks passed!")
    print("\n🚀 You're ready to run the RAG pipeline!")
    print("\nTry:")
    print("  cd src")
    print("  python rag_pipeline.py")
    print()

sys.exit(0)
