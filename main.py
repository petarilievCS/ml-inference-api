from fastapi import FastAPI, Request, HTTPException, status, Body
from fastapi.security import OAuth2PasswordBearer
from cache import client
from rate_limiter import is_rate_limited
from prometheus_fastapi_instrumentator import Instrumentator
from contextlib import asynccontextmanager
from pydantic import BaseModel

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

app = FastAPI(
    title="ML Inference API",
    description="A FastAPI app for ML inference with caching and rate limiting",
    version="1.0.0",
    contact={
        "name": "Petar Iliev",
        "email": "petariliev2002@gmail.com",
    }
)
auth = OAuth2PasswordBearer(tokenUrl="token")
instrumentator = Instrumentator().instrument(app)

class PredictRequest(BaseModel):
    text: str

class PredictBatchRequest(BaseModel):
    prompts: list[str]

@asynccontextmanager
async def lifespan(app: FastAPI):
    instrumentator.expose(app)
    yield

app = FastAPI(lifespan=lifespan)

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict")
async def predict(request: Request, body: PredictRequest = Body(...)):
    """
    Predict the classification for a single text prompt.

    This endpoint accepts a JSON request containing a single text prompt and returns 
    the classification result. It checks the cache first, and if the result is not found, 
    it performs a real-time model inference.

    Request Body:
    - **text** (str): The text prompt to classify.

    Returns:
    - **dict**: The classification result, either from the cache or a real-time inference.

    Raises:
    - **HTTPException (429)**: If the user exceeds the rate limit.
    - **HTTPException (500)**: If an unexpected error occurs.
    """
    start_time = time.time()
    user_ip = request.client.host

    logging.info(f"Received request from {user_ip}")

    try:
        check_rate(request)
        prompt = body.text
        
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
async def predict_batch(request: Request, body: PredictBatchRequest = Body(...)):
    """
    Predict classifications for multiple text prompts in a batch.

    This endpoint accepts a JSON request containing multiple text prompts 
    and returns a dictionary mapping each prompt to its classification result. 
    It uses Redis pipelines for batch caching to improve performance.

    Request Body:
    - **prompts** (list of str): A list of text prompts to classify.

    Returns:
    - **dict**: A dictionary mapping each prompt to its classification result, 
      either from the cache or a real-time inference.

    Raises:
    - **HTTPException (429)**: If the user exceeds the rate limit.
    - **HTTPException (500)**: If an unexpected error occurs.
    """
    start_time = time.time()
    user_ip = request.client.host
    logging.info(f"Received batch request from {user_ip}")

    try:
        check_rate(request)

        prompts = body.prompts
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
    """
    Invalidate all cache keys with a specific prefix.

    This endpoint deletes all cache keys that match the given prefix, 
    freeing up memory and removing potentially outdated data.

    Path Parameters:
    - **prefix** (str): The prefix for the keys to invalidate.

    Returns:
    - **dict**: A confirmation message indicating the number of keys invalidated.

    Raises:
    - **HTTPException (429)**: If the user exceeds the rate limit.
    - **HTTPException (500)**: If an unexpected error occurs.
    """
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