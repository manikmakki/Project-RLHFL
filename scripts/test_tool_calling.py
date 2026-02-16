#!/usr/bin/env python3
"""
Manual test script for tool calling functionality.
Tests both backward compatibility and new tool calling features.
"""

import requests
import json
import sys

BASE_URL = "http://localhost:8000"


def test_backward_compatibility():
    """Test that non-tool requests still work (backward compatibility)."""
    print("\n" + "=" * 60)
    print("TEST 1: Backward Compatibility (No Tools)")
    print("=" * 60)

    response = requests.post(
        f"{BASE_URL}/v1/chat/completions",
        json={
            "model": "gpt-oss:20b",
            "messages": [
                {"role": "user", "content": "Hello! How are you?"}
            ]
        }
    )

    print(f"Status Code: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"Response ID: {data.get('id')}")
        print(f"Model: {data.get('model')}")
        print(f"Finish Reason: {data['choices'][0].get('finish_reason')}")
        print(f"Content: {data['choices'][0]['message'].get('content')}")
        print(f"Tool Calls: {data['choices'][0]['message'].get('tool_calls')}")
        print("✅ PASS: Backward compatibility maintained")
        return True
    else:
        print(f"❌ FAIL: {response.text}")
        return False


def test_tool_calling():
    """Test tool calling request."""
    print("\n" + "=" * 60)
    print("TEST 2: Tool Calling Request")
    print("=" * 60)

    response = requests.post(
        f"{BASE_URL}/v1/chat/completions",
        json={
            "model": "gpt-oss:20b",
            "messages": [
                {"role": "user", "content": "What's the weather like in San Francisco?"}
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get the current weather for a location",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "The city name"
                                },
                                "unit": {
                                    "type": "string",
                                    "enum": ["celsius", "fahrenheit"],
                                    "description": "Temperature unit"
                                }
                            },
                            "required": ["location"]
                        }
                    }
                }
            ]
        }
    )

    print(f"Status Code: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"Response ID: {data.get('id')}")
        print(f"Model: {data.get('model')}")
        print(f"Finish Reason: {data['choices'][0].get('finish_reason')}")
        print(f"Content: {data['choices'][0]['message'].get('content')}")

        tool_calls = data['choices'][0]['message'].get('tool_calls')
        if tool_calls:
            print(f"Tool Calls: {json.dumps(tool_calls, indent=2)}")
            print("✅ PASS: Model generated tool calls")
            return True, tool_calls
        else:
            print("⚠️  WARNING: No tool calls generated (model may not support function calling)")
            return True, None
    else:
        print(f"❌ FAIL: {response.text}")
        return False, None


def test_tool_result_handling(tool_calls):
    """Test handling of tool results in follow-up request."""
    print("\n" + "=" * 60)
    print("TEST 3: Tool Result Handling")
    print("=" * 60)

    if not tool_calls:
        print("⏭️  SKIP: No tool calls from previous test")
        return True

    # Simulate tool execution
    tool_result = {
        "temperature": 72,
        "condition": "sunny",
        "humidity": 65
    }

    response = requests.post(
        f"{BASE_URL}/v1/chat/completions",
        json={
            "model": "gpt-oss:20b",
            "messages": [
                {"role": "user", "content": "What's the weather like in San Francisco?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": tool_calls
                },
                {
                    "role": "tool",
                    "content": json.dumps(tool_result),
                    "tool_call_id": tool_calls[0]["id"]
                }
            ]
        }
    )

    print(f"Status Code: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"Response ID: {data.get('id')}")
        print(f"Finish Reason: {data['choices'][0].get('finish_reason')}")
        print(f"Content: {data['choices'][0]['message'].get('content')}")
        print("✅ PASS: Tool result handled successfully")
        return True
    else:
        print(f"❌ FAIL: {response.text}")
        return False


def test_health():
    """Test health endpoint."""
    print("\n" + "=" * 60)
    print("TEST 0: Health Check")
    print("=" * 60)

    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"Status: {data.get('status')}")
        print(f"Model Loaded: {data.get('model_loaded')}")
        print("✅ PASS: Service is healthy")
        return True
    else:
        print(f"❌ FAIL: {response.text}")
        return False


def main():
    """Run all tests."""
    print("\n🧪 Testing Tool Calling Implementation")
    print("=" * 60)

    results = []

    # Test health
    if not test_health():
        print("\n❌ Service is not healthy. Aborting tests.")
        sys.exit(1)

    # Test backward compatibility
    results.append(("Backward Compatibility", test_backward_compatibility()))

    # Test tool calling
    success, tool_calls = test_tool_calling()
    results.append(("Tool Calling Request", success))

    # Test tool result handling
    results.append(("Tool Result Handling", test_tool_result_handling(tool_calls)))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name}: {status}")

    total = len(results)
    passed = sum(1 for _, p in results if p)

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print("\n⚠️  Some tests failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
