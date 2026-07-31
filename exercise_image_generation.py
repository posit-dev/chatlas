from chatlas import ChatOpenAI

chat = ChatOpenAI(model="gpt-5")

chat.chat(
    "Generate a simple watercolor illustration of a lighthouse at sunrise.",
    echo="output",
    kwargs={
        "tools": [
            {
                "type": "image_generation",
                "size": "1024x1024",
            }
        ],
        "tool_choice": "required",
    },
)
