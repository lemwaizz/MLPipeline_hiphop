# locustfile.py

from locust import HttpUser, task, between
from pathlib import Path

SAMPLE_AUDIO = Path("data/test_example.mp3")  # put a small test file here


class EraClassifierUser(HttpUser):
    wait_time = between(1, 5)

    @task
    def predict_audio(self):
        with open(SAMPLE_AUDIO, "rb") as f:
            files = {"file": (SAMPLE_AUDIO.name, f.read())}
        self.client.post("/predict-audio", files=files)

