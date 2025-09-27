import os


def _typed_fallback(prompt: str) -> str:
    try:
        return input(prompt)
    except EOFError:
        return "fetch tissue"


try:
    import speech_recognition as sr  # type: ignore
except Exception:
    sr = None  # type: ignore


def stt_to_text() -> str:
    """Return a text command using microphone when available, otherwise prompt typing.

    Environment variables:
    - STT_METHOD: "mic" or "type" (default: "mic" if SpeechRecognition is installed else "type")
    - STT_MIC_DEVICE_INDEX: optional int device index for microphone
    - STT_TIMEOUT_S: max seconds to wait for phrase start (default: 5)
    - STT_PHRASE_TIME_LIMIT_S: max seconds for phrase (default: 10)
    - STT_LANG: BCP-47 language code for recognition (default: en-US)
    """

    method = os.environ.get("STT_METHOD", "mic" if sr is not None else "type").lower()
    if method != "mic" or sr is None:
        return _typed_fallback("Enter prompt now (e.g., 'fetch tissue'): ")

    device_index = None
    idx_str = os.environ.get("STT_MIC_DEVICE_INDEX")
    if idx_str not in (None, ""):
        try:
            device_index = int(idx_str)
        except ValueError:
            device_index = None

    try:
        timeout_s = float(os.environ.get("STT_TIMEOUT_S", "5"))
    except ValueError:
        timeout_s = 5.0

    try:
        phrase_time_limit_s = float(os.environ.get("STT_PHRASE_TIME_LIMIT_S", "10"))
    except ValueError:
        phrase_time_limit_s = 10.0

    recognizer = sr.Recognizer() if sr is not None else None  # type: ignore[assignment]
    if recognizer is None:
        return _typed_fallback("Enter prompt now (e.g., 'fetch tissue'): ")

    recognizer.dynamic_energy_threshold = True

    try:
        with sr.Microphone(device_index=device_index) as source:  # type: ignore[attr-defined]
            print("[voice] Listening... (Ctrl+C to cancel)")
            try:
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
            except Exception:
                pass
            audio = recognizer.listen(
                source, timeout=timeout_s, phrase_time_limit=phrase_time_limit_s
            )

        lang = os.environ.get("STT_LANG", "en-US")
        try:
            text = recognizer.recognize_google(audio, language=lang)
            print(f"[voice] Heard: {text}")
            return text
        except sr.UnknownValueError:  # type: ignore[attr-defined]
            print("[voice] Could not understand audio; fallback to typing.")
            return _typed_fallback("Enter prompt (fallback): ")
        except sr.RequestError as e:  # type: ignore[attr-defined]
            print(f"[voice] STT request error: {e}; fallback to typing.")
            return _typed_fallback("Enter prompt (fallback): ")

    except KeyboardInterrupt:
        print("\n[voice] Cancelled. Fallback to typing.")
        return _typed_fallback("Enter prompt (fallback): ")
    except Exception as e:
        print(f"[voice] Microphone error: {e}; fallback to typing.")
        return _typed_fallback("Enter prompt (fallback): ")


