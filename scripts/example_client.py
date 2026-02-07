#!/usr/bin/env python3
"""
Example client demonstrating how to use the Project RLHFL API.

This shows various ways to interact with the system and how sentiment
is automatically inferred from conversation patterns.
"""

from openai import OpenAI
import time


def main():
    # Initialize client pointing to local API
    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="not-needed"  # No API key required for local
    )
    
    print("=" * 80)
    print("Project RLHFL - EXAMPLE CLIENT")
    print("=" * 80)
    print()
    
    # Example 1: Basic conversation
    print("Example 1: Basic Conversation")
    print("-" * 40)
    
    response = client.chat.completions.create(
        model="local-llm",
        messages=[
            {"role": "user", "content": "What is the capital of France?"}
        ]
    )
    
    answer = response.choices[0].message.content
    print(f"User: What is the capital of France?")
    print(f"Assistant: {answer}")
    print()
    
    # Example 2: Positive feedback (will be learned)
    print("Example 2: Positive Feedback")
    print("-" * 40)
    
    response = client.chat.completions.create(
        model="local-llm",
        messages=[
            {"role": "user", "content": "Explain quantum computing in simple terms"}
        ]
    )
    
    answer = response.choices[0].message.content
    print(f"User: Explain quantum computing in simple terms")
    print(f"Assistant: {answer[:200]}...")
    print()
    
    # Simulate positive feedback
    print("User: Thanks! That's a great explanation!")
    response = client.chat.completions.create(
        model="local-llm",
        messages=[
            {"role": "user", "content": "Explain quantum computing in simple terms"},
            {"role": "assistant", "content": answer},
            {"role": "user", "content": "Thanks! That's a great explanation!"}
        ]
    )
    print(f"Assistant: {response.choices[0].message.content}")
    print("→ System automatically infers POSITIVE sentiment and increases weight")
    print()
    
    # Example 3: Explicit instruction (golden example)
    print("Example 3: Explicit Instruction (High Value)")
    print("-" * 40)
    
    response = client.chat.completions.create(
        model="local-llm",
        messages=[
            {"role": "user", "content": "Remember to always format code examples with syntax highlighting and include comments explaining each section."}
        ]
    )
    
    print("User: Remember to always format code examples with syntax highlighting...")
    print(f"Assistant: {response.choices[0].message.content}")
    print("→ System detects 'Remember to always' as HIGH-VALUE instruction")
    print("→ Marked as GOLDEN EXAMPLE for replay buffer")
    print()
    
    # Example 4: Streaming response
    print("Example 4: Streaming Response")
    print("-" * 40)
    
    print("User: Tell me a short story about a robot")
    print("Assistant: ", end="", flush=True)
    
    stream = client.chat.completions.create(
        model="local-llm",
        messages=[
            {"role": "user", "content": "Tell me a short story about a robot"}
        ],
        stream=True
    )
    
    full_response = ""
    for chunk in stream:
        if chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            print(content, end="", flush=True)
            full_response += content
    
    print()
    print("→ Streaming complete, interaction stored in memory")
    print()
    
    # Example 5: Check training stats
    print("Example 5: Training Statistics")
    print("-" * 40)
    
    import requests
    stats_response = requests.get("http://localhost:8000/v1/training/stats")
    
    if stats_response.status_code == 200:
        stats = stats_response.json()
        print(f"Total interactions: {stats['total_interactions']}")
        print(f"New interactions since last training: {stats['new_interactions_since_last_training']}")
        print(f"Hours since last interaction: {stats['hours_since_last_interaction']:.1f}")
        print(f"Days since last training: {stats['days_since_last_training']:.1f}")
        print()
        
        print("Training will trigger when:")
        print("  - 50+ new interactions accumulated, OR")
        print("  - 24+ hours of inactivity with 10+ interactions, OR")
        print("  - 7+ days since last training")
    else:
        print("Failed to get training stats")
    print()
    
    # Example 6: Manual training trigger
    print("Example 6: Manual Training Trigger")
    print("-" * 40)
    print("You can manually trigger training at any time:")
    print("  curl -X POST http://localhost:8000/v1/training/trigger")
    print()
    
    print("=" * 80)
    print("Examples complete!")
    print()
    print("The system is now learning from these interactions.")
    print("Continue chatting normally - sentiment is inferred automatically:")
    print("  - 'Thanks!' / 'Perfect!' → Positive feedback")
    print("  - 'No, wrong' / 'Try again' → Negative feedback")
    print("  - 'Always do X' / 'Remember to Y' → High-value instruction")
    print("=" * 80)


if __name__ == "__main__":
    main()
