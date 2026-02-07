#!/usr/bin/env python3
"""
Health check script for the Project RLHFL system.
"""

import sys
import requests
import json
from typing import Dict, Any


def check_api_health() -> tuple[bool, Dict[str, Any]]:
    """Check if API service is healthy."""
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, {"error": f"Status code: {response.status_code}"}
    except requests.exceptions.RequestException as e:
        return False, {"error": str(e)}


def check_models_endpoint() -> tuple[bool, Dict[str, Any]]:
    """Check if models endpoint is working."""
    try:
        response = requests.get("http://localhost:8000/v1/models", timeout=5)
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, {"error": f"Status code: {response.status_code}"}
    except requests.exceptions.RequestException as e:
        return False, {"error": str(e)}


def check_training_stats() -> tuple[bool, Dict[str, Any]]:
    """Check training statistics."""
    try:
        response = requests.get("http://localhost:8000/v1/training/stats", timeout=5)
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, {"error": f"Status code: {response.status_code}"}
    except requests.exceptions.RequestException as e:
        return False, {"error": str(e)}


def test_completion() -> tuple[bool, Dict[str, Any]]:
    """Test a simple completion."""
    try:
        payload = {
            "model": "local-llm",
            "messages": [
                {"role": "user", "content": "Say 'Hello, I am working!' and nothing else."}
            ],
            "max_tokens": 50,
            "temperature": 0.1
        }
        
        response = requests.post(
            "http://localhost:8000/v1/chat/completions",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            if "choices" in data and len(data["choices"]) > 0:
                message = data["choices"][0]["message"]["content"]
                return True, {"response": message}
            else:
                return False, {"error": "No choices in response"}
        else:
            return False, {"error": f"Status code: {response.status_code}"}
            
    except requests.exceptions.RequestException as e:
        return False, {"error": str(e)}


def main():
    """Run all health checks."""
    print("=" * 80)
    print("Project RLHFL - HEALTH CHECK")
    print("=" * 80)
    print()
    
    all_passed = True
    
    # Check 1: API Health
    print("1. Checking API health...")
    passed, data = check_api_health()
    if passed:
        print("   ✓ API is healthy")
        print(f"   - Model loaded: {data.get('model_loaded')}")
        print(f"   - Memory connected: {data.get('memory_connected')}")
        print(f"   - GPU available: {data.get('gpu_available')}")
    else:
        print(f"   ✗ API health check failed: {data.get('error')}")
        all_passed = False
    print()
    
    # Check 2: Models Endpoint
    print("2. Checking models endpoint...")
    passed, data = check_models_endpoint()
    if passed:
        print("   ✓ Models endpoint working")
        if "data" in data:
            print(f"   - Available models: {len(data['data'])}")
    else:
        print(f"   ✗ Models endpoint failed: {data.get('error')}")
        all_passed = False
    print()
    
    # Check 3: Training Stats
    print("3. Checking training stats...")
    passed, data = check_training_stats()
    if passed:
        print("   ✓ Training stats available")
        print(f"   - Total interactions: {data.get('total_interactions')}")
        print(f"   - New interactions: {data.get('new_interactions_since_last_training')}")
    else:
        print(f"   ✗ Training stats failed: {data.get('error')}")
        all_passed = False
    print()
    
    # Check 4: Test Completion
    print("4. Testing completion generation...")
    passed, data = check_completion()
    if passed:
        print("   ✓ Completion generation working")
        print(f"   - Response: {data.get('response', '')[:100]}...")
    else:
        print(f"   ✗ Completion test failed: {data.get('error')}")
        all_passed = False
    print()
    
    # Summary
    print("=" * 80)
    if all_passed:
        print("HEALTH CHECK PASSED - All systems operational")
        print("=" * 80)
        return 0
    else:
        print("HEALTH CHECK FAILED - Some systems not operational")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    sys.exit(main())
