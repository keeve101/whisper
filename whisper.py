import speech_recognition as sr

from gemini import GEMINI_API
from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.llms.llama_cpp import LlamaCPP

from datetime import datetime, timedelta
from queue import Queue
from time import sleep
from whisper_cpp_python import run
import wave

WHISPER_MODEL = (
    "C:\\Users\\keith\\Desktop\\repos\\whisper.cpp\\models\\ggml-base.en.bin"
)
LLM_MODEL = "C:\\Users\\keith\\Desktop\\repos\\whisper\\models\\chat\\Meta-Llama-3-8B-Instruct-Q8_0.gguf"


def main():
    llm = LlamaCPP(model_path=LLM_MODEL, verbose=False)
    chat_engine = SimpleChatEngine.from_defaults(llm=llm)

    # The last time a recording was retrieved from the queue.
    phrase_time = None
    # Thread safe Queue for passing data from the threaded recording callback.
    data_queue = Queue()
    # We use SpeechRecognizer to record our audio because it has a nice feature where it can detect when speech ends.
    recorder = sr.Recognizer()
    recorder.energy_threshold = 1000
    # Definitely do this, dynamic energy compensation lowers the energy threshold dramatically to a point where the SpeechRecognizer never stops recording.
    recorder.dynamic_energy_threshold = False

    source = sr.Microphone(sample_rate=16000)

    record_timeout = 2
    phrase_timeout = 3

    transcription = [""]

    with source:
        recorder.adjust_for_ambient_noise(source)

    def record_callback(_, audio: sr.AudioData) -> None:
        """
        Threaded callback function to receive audio data when recordings finish.
        audio: An AudioData containing the recorded bytes.
        """
        # Grab the raw bytes and push it into the thread safe queue.
        data = audio.get_raw_data()
        data_queue.put(data)

    # Create a background thread that will pass us raw audio bytes.
    # We could do this manually but SpeechRecognizer provides a nice helper.
    recorder.listen_in_background(
        source, record_callback, phrase_time_limit=record_timeout
    )

    while True:
        try:
            now = datetime.now()
            # Pull raw recorded audio from the queue.
            if not data_queue.empty():
                phrase_complete = False
                # If enough time has passed between recordings, consider the phrase complete.
                # Clear the current working audio buffer to start over with the new data.
                if phrase_time and now - phrase_time > timedelta(
                    seconds=phrase_timeout
                ):
                    phrase_complete = True
                # This is the last time we received new audio data from the queue.
                phrase_time = now

                # Combine audio data from queue
                audio_data = b"".join(data_queue.queue)
                data_queue.queue.clear()

                sample_width = 2  # 2 bytes for 16-bit audio
                n_channels = 1  # Mono audio
                sample_rate = 16000  # 16 bit sample rate

                with wave.open("output.wav", "wb") as wav_file:
                    wav_file.setnchannels(n_channels)
                    wav_file.setsampwidth(sample_width)
                    wav_file.setframerate(sample_rate)
                    wav_file.writeframes(audio_data)

                transcription, error = run(WHISPER_MODEL, "output.wav")

                user_message = f"User: {transcription.strip()}"
                print(user_message)
                response = chat_engine.chat(transcription)

                response_message = f"Assistant: {response}"
                print(response_message)
            else:
                # Infinite loops are bad for processors, must sleep.
                sleep(0.25)
        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    main()
