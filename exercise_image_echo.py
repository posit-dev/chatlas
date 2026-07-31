from chatlas import ChatOpenAI, content_image_url


def remote_image() -> object:
    "Return a remote PNG image."
    return content_image_url("https://httpbin.org/image/png")


chat = ChatOpenAI()
chat.register_tool(remote_image)

chat.chat(
    "Call the remote_image tool, then briefly describe the image.",
    echo="output",
)
