from fastapi import FastAPI, Request, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from cache import client
from rate_limiter import is_rate_limited

import logging
import cache
import model

logging.basicConfig(level=logging.INFO)

app = FastAPI()
auth = OAuth2PasswordBearer(tokenUrl="token")

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict")
async def predict(request: Request):
    check_rate(request)
    user_ip = request.client.host

    data = await request.json()
    prompt = data["text"]
    
    key = cache.generate_hash(prompt, user_ip)
    result = cache.get(key)
    if result:
        return result
    
    result = model.classify(prompt)
    cache.set(key, result)

    return result

@app.post("/predict_batch")
async def predict_batch(request: Request):
    check_rate(request)
    user_ip = request.client.host

    data = await request.json()
    prompts = data["prompts"]

    results = {}
    with client.pipeline() as pipe:
        for prompt in prompts:
            key = cache.generate_hash(prompt, user_ip)
            result = cache.get(key)
            if result == None:
                result = model.classify(prompt)
                cache.batch_set(pipe, key, result)
            results[prompt] = result
        pipe.execute()

    return results

@app.delete("/invalidate/{prefix}", status_code=status.HTTP_204_NO_CONTENT)
async def invalidate(prefix: str, request: Request):
    check_rate(request)
    result = cache.invalidate_prefix(prefix)
    return {"message": f"Invalidated {result} keys with prefix '{prefix}'"} 

# Helpers

def check_rate(request: Request):
    user_ip = request.client.host
    if is_rate_limited(user_ip):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please try again later.",
            headers={"Retry-After": "60"}
            )