from locust import HttpUser, between, task
import random
import string

class MLInferenceUser(HttpUser):
    wait_time = between(1, 5)

    @task(1)
    def predict(self):
        prompt = "".join(random.choices(string.ascii_lowercase, k=10))
        self.client.post("/predict", json={"text": prompt})
    
    @task(2)
    def predict_batch(self):
        prompts = ["".join(random.choices(string.ascii_lowercase, k=10)) for _ in range(5)]
        self.client.post("/predict_batch", json={"prompts": prompts})
    
    @task(0.1)
    def invalidate(self):
        prefix = "".join(random.choices(string.ascii_lowercase, k=5))
        self.client.delete(f"/invalidate/{prefix}")