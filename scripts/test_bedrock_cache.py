"""
Quick test: can ChatBedrockAnthropic work with non-Anthropic models?
"""

from chatlas import ChatBedrockAnthropic

for model in ["amazon.nova-lite-v1:0", "us.amazon.nova-lite-v1:0", "meta.llama3-1-8b-instruct-v1:0"]:
    print(f"\n--- Testing model: {model} ---")
    try:
        chat = ChatBedrockAnthropic(model=model)
        response = chat.chat("What is 1 + 1?")
        print(f"  Success: {response}")
    except Exception as e:
        print(f"  Error ({type(e).__name__}): {e}")
