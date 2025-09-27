def stt_to_text() -> str:
    try:
        # Explicit prompt so it's obvious in terminal
        return input("Enter prompt now (e.g., 'fetch tissue'): ")
    except EOFError:
        return "fetch tissue"


