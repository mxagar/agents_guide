"""
Comprehensive Langfuse Integration Diagnostic Tool
Checks all common issues that prevent traces from appearing
"""

import os
import sys
import time
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("="*70)
print("LANGFUSE INTEGRATION DIAGNOSTIC TOOL")
print("="*70)
print()

# ============================================================================
# 1. CHECK API CREDENTIALS
# ============================================================================
print("1️⃣  CHECKING API CREDENTIALS")
print("-" * 70)

public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
secret_key = os.getenv("LANGFUSE_SECRET_KEY")
host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

if not public_key:
    print("❌ LANGFUSE_PUBLIC_KEY not set!")
    sys.exit(1)
if not secret_key:
    print("❌ LANGFUSE_SECRET_KEY not set!")
    sys.exit(1)

print(f"✅ LANGFUSE_PUBLIC_KEY: {public_key[:20]}...")
print(f"✅ LANGFUSE_SECRET_KEY: {secret_key[:20]}...")
print(f"✅ LANGFUSE_HOST: {host}")
print()

# ============================================================================
# 2. VERIFY NETWORK CONNECTIVITY
# ============================================================================
print("2️⃣  TESTING NETWORK CONNECTIVITY")
print("-" * 70)

try:
    response = requests.get(
        f"{host}/api/public/health",
        timeout=10
    )
    if response.status_code == 200:
        print(f"✅ Successfully connected to {host}")
    else:
        print(f"⚠️  Unexpected response: {response.status_code}")
except requests.exceptions.ConnectionError:
    print(f"❌ Cannot connect to {host}")
    print("   Check your network connection and firewall settings")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)
print()

# ============================================================================
# 3. VALIDATE API CREDENTIALS
# ============================================================================
print("3️⃣  VALIDATING API CREDENTIALS")
print("-" * 70)

try:
    response = requests.get(
        f"{host}/api/public/projects",
        auth=(public_key, secret_key),
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
        sys.exit(1)
    else:
        print(f"⚠️  Unexpected response: {response.status_code}")
        print(f"   Response: {response.text}")
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)
print()

# ============================================================================
# 4. TEST LANGFUSE SDK INTEGRATION
# ============================================================================
print("4️⃣  TESTING LANGFUSE SDK INTEGRATION")
print("-" * 70)

try:
    from langfuse import Langfuse
    
    # Initialize client
    client = Langfuse(
        public_key=public_key,
        secret_key=secret_key,
        host=host,
        debug=True  # Enable debug logging
    )
    
    print("✅ Langfuse SDK initialized successfully")
    
    # Create a test trace
    trace_id = f"diagnostic-test-{int(time.time())}"
    print(f"✅ Creating test trace: {trace_id}")
    
    trace = client.trace(
        id=trace_id,
        name="Diagnostic Test Trace",
        metadata={"source": "diagnostic_script"}
    )
    
    # Add a span
    span = trace.span(
        name="test_operation",
        metadata={"test": True}
    )
    span.end()
    
    # Add a generation
    generation = trace.generation(
        name="test_generation",
        model="test-model",
        input="Hello, world!",
        output="This is a test response"
    )
    
    print("✅ Test trace created with span and generation")
    
except ImportError:
    print("❌ Langfuse SDK not installed!")
    print("   Install with: pip install langfuse")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error initializing Langfuse: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
print()

# ============================================================================
# 5. TEST MANUAL FLUSHING
# ============================================================================
print("5️⃣  TESTING MANUAL FLUSH")
print("-" * 70)

print("⏳ Flushing events to Langfuse...")
try:
    client.flush()
    print("✅ Flush completed successfully")
    print("   Note: Events are sent asynchronously, may take a few seconds to appear")
except Exception as e:
    print(f"❌ Flush error: {e}")
    import traceback
    traceback.print_exc()
print()

# ============================================================================
# 6. VERIFY TRACE IN API
# ============================================================================
print("6️⃣  VERIFYING TRACE IN API")
print("-" * 70)

print("⏳ Waiting 3 seconds for trace to be processed...")
time.sleep(3)

try:
    # Try to fetch the trace we just created
    response = requests.get(
        f"{host}/api/public/traces/{trace_id}",
        auth=(public_key, secret_key),
        timeout=10
    )
    
    if response.status_code == 200:
        print(f"✅ TEST SUCCESSFUL! Trace found in Langfuse")
        trace_data = response.json()
        print(f"   Trace name: {trace_data.get('name')}")
        print(f"   Timestamp: {trace_data.get('timestamp')}")
        print()
        print("🎉 Your Langfuse integration is working correctly!")
        print(f"   View in UI: {host}/project/{projects['data'][0]['id']}/traces/{trace_id}")
    elif response.status_code == 404:
        print("⚠️  Trace not found yet (404)")
        print("   This might be normal - traces can take a few seconds to appear")
        print("   Try checking the Langfuse UI in ~10-30 seconds")
    else:
        print(f"⚠️  Unexpected response: {response.status_code}")
        print(f"   Response: {response.text}")
except Exception as e:
    print(f"⚠️  Could not verify trace: {e}")
    print("   This doesn't mean it failed - check the UI")
print()

# ============================================================================
# 7. CONFIGURATION RECOMMENDATIONS
# ============================================================================
print("7️⃣  CONFIGURATION RECOMMENDATIONS")
print("-" * 70)

print("For short-lived scripts (like this diagnostic tool):")
print("  • Always call client.flush() before script exits")
print("  • Consider using client.flush_at=1 for immediate sends")
print()
print("For high-throughput applications:")
print("  • Adjust flush_at (default: 15) - events per batch")
print("  • Adjust flush_interval (default: 1.0s) - time between flushes")
print()
print("For debugging:")
print("  • Set debug=True in Langfuse() constructor")
print("  • Check application logs for errors")
print()
print("Next steps:")
print("  1. Check the Langfuse UI for the test trace created above")
print(f"     URL: {host}/project/{projects['data'][0]['id']}")
print("  2. If you see it, your integration is working!")
print("  3. If not, check that traces are being created AND flushed in your app")
print()
print("="*70)
