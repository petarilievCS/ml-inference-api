from fastapi import FastAPI, Request, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from cache import client
from rate_limiter import is_rate_limited
from prometheus_fastapi_instrumentator import Instrumentator
from contextlib import asynccontextmanager

import logging
import cache
import model
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("api.log"),
        logging.StreamHandler()
    ]
)

app = FastAPI()
auth = OAuth2PasswordBearer(tokenUrl="token")

@asynccontextmanager
async def lifespan(app: FastAPI):
    Instrumentator().instrument(app).expose(app)
    yield

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict")
async def predict(request: Request):
    start_time = time.time()
    user_ip = request.client.host

    logging.info(f"Received request from {user_ip}")

    try:
        check_rate(request)
        data = await request.json()
        prompt = data["text"]
        
        key = cache.generate_hash(prompt, user_ip)
        logging.info(f"Generated key '{key}' for prompt from {user_ip}")

        result = cache.get(key)
        if result:
            logging.info(f"Cache hit for key '{key}' from {user_ip}")
            response_time = time.time() - start_time
            logging.info(f"Response time: {response_time:.4f} seconds")
            return result
        
        result = model.classify(prompt)
        cache.set(key, result)

        logging.info(f"Cache miss for key '{key}' from {user_ip}")
        response_time = time.time() - start_time
        logging.info(f"Response time: {response_time:.4f} seconds")

        return result

    except Exception as e:
        logging.error(f"Error processing request from {user_ip}: {e}")
        raise
    
@app.post("/predict_batch")
async def predict_batch(request: Request):
    start_time = time.time()
    user_ip = request.client.host
    logging.info(f"Received batch request from {user_ip}")

    try:
        check_rate(request)

        data = await request.json()
        prompts = data["prompts"]
        logging.info(f"Processing {len(prompts)} prompts from {user_ip}")

        results = {}
        with client.pipeline() as pipe:
            for prompt in prompts:
                key = cache.generate_hash(prompt, user_ip)
                logging.info(f"Generated key '{key}' for prompt from {user_ip}")

                result = cache.get(key)
                if result is None:
                    logging.info(f"Cache miss for key '{key}' from {user_ip}")
                    result = model.classify(prompt)
                    cache.batch_set(pipe, key, result)
                else:
                    logging.info(f"Cache hit for key '{key}' from {user_ip}")

                results[prompt] = result

            pipe.execute()

        response_time = time.time() - start_time
        logging.info(f"Batch request from {user_ip} completed in {response_time:.4f} seconds")

        return results

    except Exception as e:
        logging.error(f"Error processing batch request from {user_ip}: {e}")
        raise

@app.delete("/invalidate/{prefix}", status_code=status.HTTP_204_NO_CONTENT)
async def invalidate(prefix: str, request: Request):
    start_time = time.time()
    user_ip = request.client.host
    logging.info(f"Received invalidate request from {user_ip} for prefix '{prefix}'")

    try:
        check_rate(request)
        result = cache.invalidate_prefix(prefix)
        
        response_time = time.time() - start_time
        logging.info(f"Invalidated {result} keys with prefix '{prefix}' from {user_ip} in {response_time:.4f} seconds")
        
        return {"message": f"Invalidated {result} keys with prefix '{prefix}'"}

    except Exception as e:
        logging.error(f"Error processing invalidate request from {user_ip} for prefix '{prefix}': {e}")
        raise

# Helpers
def check_rate(request: Request):
    user_ip = request.client.host
    logging.info(f"Checking rate limit for {user_ip}")

    if is_rate_limited(user_ip):
        logging.warning(f"Rate limit exceeded for {user_ip}")
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please try again later.",
            headers={"Retry-After": "60"}
        )