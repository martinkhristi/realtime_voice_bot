import assemblyai as aai
from elevenlabs import stream
from elevenlabs.client import ElevenLabs
from groq import Groq
import os

class AI_Assistant:
    def __init__(self):
        aai.settings.api_key = "assemblyai api key"
        self.groq_client = Groq(api_key=os.environ.get('groq api key'))
        self.elevenlabs_api_key = "elevenlabs api key"

        self.elevenlabs_client = ElevenLabs(api_key=self.elevenlabs_api_key)

        self.transcriber = None

        self.interaction = [
            {"role": "system", "content": "your helpful guide for Copenhagen, Denmark. Ask me anything about planning your trip, from must-see attractions to food recommendations, and I'll make it easy and enjoyable! 😊."},
        ]

    def stop_transcription(self):
        if self.transcriber:
            self.transcriber.close()
            self.transcriber = None

    def on_open(self, session_opened: aai.RealtimeSessionOpened):
        print("Session ID:", session_opened.session_id)
        return

    def on_error(self, error: aai.RealtimeError):
        print("An error occurred:", error)
        return

    def on_close(self):
        print("Closing Session")
        return

    def on_data(self, transcript: aai.RealtimeTranscript):
        if not transcript.text:
            return
        if isinstance(transcript, aai.RealtimeFinalTranscript):
            self.generate_ai_response(transcript)
        else:
            print(transcript.text, end="\r")

    def start_transcription(self):
        self.transcriber = aai.RealtimeTranscriber(
            sample_rate=16000,
            on_data=self.on_data,
            on_error=self.on_error,
            on_open=self.on_open,
            on_close=self.on_close,
            end_utterance_silence_threshold=1000
        )

        self.transcriber.connect()
        microphone_stream = aai.extras.MicrophoneStream(sample_rate=16000)
        self.transcriber.stream(microphone_stream)

    def generate_ai_response(self, transcript):
        self.stop_transcription()

        self.interaction.append({"role": "user", "content": transcript.text})
        print(f"\nTourist: {transcript.text}", end="\r\n")

        response = self.groq_client.chat.completions.create(
            model="llama-3.2-3b-preview",
            messages=self.interaction
        )

        ai_response = response.choices[0].message.content

        self.generate_audio(ai_response)

        self.start_transcription()
        print(f"\nReal-time transcription: ", end="\r\n")

    def generate_audio(self, text):
        self.interaction.append({"role": "assistant", "content": text})
        print(f"\nAI Guide: {text}")

        audio_stream = self.elevenlabs_client.generate(
            text=text,
            voice="Rachel",
            stream=True
        )

        stream(audio_stream)

greeting = "Thank you for calling Copenhagen Travel Guide. My name is Rachel, how may I assist you?"
ai_assistant = AI_Assistant()
ai_assistant.generate_audio(greeting)
ai_assistant.start_transcription()
