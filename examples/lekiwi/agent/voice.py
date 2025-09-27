def stt_to_text() -> str:
    try:
        return input("Say command (type for demo): ")
    except EOFError:
        return "fetch tissue"


