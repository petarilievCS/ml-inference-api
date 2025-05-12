import redis
import json
import hashlib

client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

def generate_hash(key: str, user: str):
    model = "model"
    hash = hashlib.sha256(key.encode()).hexdigest()
    return f"model2:{user}:{hash}"

def get(key: str):
    result = client.get(key)
    return json.loads(result) if result else None

def set(key: str, value: dict, ttl: int = 3600):
    client.set(key, json.dumps(value), ex=ttl)

def batch_set(pipeline, key: str, value: dict, ttl: int = 3600):
    pipeline.set(key, json.dumps(value), ex=ttl)

def invalidate(key: str):
    client.delete(key)

def invalidate_prefix(prefix: str):
    with client.pipeline() as pipe:
        for key in client.scan_iter(f"{prefix}:*"):
            pipe.delete(key)
        pipe.execute()
