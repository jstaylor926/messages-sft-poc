import requests
import os

class InferenceService:
    def __init__(self):
        self.ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
        self.model = os.getenv("OLLAMA_MODEL", "llama3")

    def _call_ollama(self, messages: list) -> str:
        try:
            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False
                }
            )
            response.raise_for_status()
            return response.json().get("message", {}).get("content", "")
        except Exception as e:
            print(f"Error calling Ollama: {e}")
            return f"Error calling Ollama: {str(e)}"

    def generate_lora(self, subject: str, body: str) -> str:
        # Using the user's provided example as a few-shot prompt to simulate the style transfer
        messages = [
            {
                "role": "system",
                "content": "You are rewriting a LinkedIn Message to fit the speach of AJ consider the speech pattern, focusing on grammar, punctuation, and sentence length of AJ. Your output should consist solely of the adapted message."
            },
            {
                "role": "user",
                "content": "I reached out to our CX Research Director and, unfortunately, we won't be offering internships this summer. The last program we ran was in 2022. I'll let you know if any suitable opportunities appear. Have you contacted the MDD? They often have strong research connections."
            },
            {
                "role": "assistant",
                "content": "I talked to our CX Research director yesterday. Unfortunately, there are no interns this summer. We did that in 2022, but apparently not this year. I will have a look if I see anything :)  Have you talked to the MDD? They also have good connections inside of research. Grateful, AJ."
            },
            {
                "role": "user",
                "content": f"Subject: {subject}\n\n{body}"
            }
        ]
        return self._call_ollama(messages)

    def generate_gpt4(self, subject: str, body: str) -> str:
        # Simulating GPT-4 with Ollama for demo purposes
        messages = [
            {
                "role": "system",
                "content": "You are a highly polished professional email assistant. Write a sophisticated reply to the following message."
            },
            {
                "role": "user",
                "content": f"Subject: {subject}\nBody: {body}\n\nReply:"
            }
        ]
        return self._call_ollama(messages)

    def generate_hybrid(self, subject: str, body: str) -> str:
        # Simulating Hybrid with Ollama for demo purposes
        messages = [
            {
                "role": "system",
                "content": "You are an expert email assistant. Write a concise and effective reply to the following message."
            },
            {
                "role": "user",
                "content": f"Subject: {subject}\nBody: {body}\n\nReply:"
            }
        ]
        return self._call_ollama(messages)
