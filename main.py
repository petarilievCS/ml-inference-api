from fastapi import FastAPI, Request, HTTPException
from cache import client
from rate_limiter import is_rate_limited

import logging
import cache
import model

logging.basicConfig(level=logging.INFO)

app = FastAPI()

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict")
async def predict(request: Request):
    check_rate(request)

    data = await request.json()
    prompt = data["text"]
    
    key = cache.generate_hash(prompt)
    result = cache.get(key)
    if result:
        return result
    
    result = model.classify(prompt)
    cache.set(key, result)

    return result

@app.post("/predict_batch")
async def predict_batch(request: Request):
    check_rate(request)

    data = await request.json()
    prompts = data["prompts"]

    results = {}
    with client.pipeline() as pipe:
        for prompt in prompts:
            key = cache.generate_hash(prompt)
            result = cache.get(key)
            if result == None:
                result = model.classify(prompt)
                cache.batch_set(pipe, key, result)
            results[prompt] = result
        pipe.execute()

    return results

def check_rate(request: Request):
    user = request.headers.get("X-User-ID")
    if is_rate_limited(user):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please try again later.",
            headers={"Retry-After": "60"}
            )