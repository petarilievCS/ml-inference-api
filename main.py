from fastapi import FastAPI, Request

import logging
import cache
import model

logging.basicConfig(level=logging.INFO)

app = FastAPI()

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict")
async def read_root(request: Request):
    data = await request.json()
    prompt = data["text"]
    
    key = cache.generate_hash(prompt)
    result = cache.get(key)
    if result:
        return result
    
    result = model.classify(prompt)
    cache.set(key, result)

    return result