import redis
import logging

LIMIT = 100
WINDOW = 60

client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

def is_rate_limited(user: str) -> bool:
    key = f'{user}:count'
    count = client.get(key)

    if count is None:
        logging.info(f"First request from user '{user}', creating rate limit counter")
        client.set(key, 1, ex=WINDOW)
        return False 
    
    if int(count) < LIMIT:
        client.incr(key)
        logging.info(f"User '{user}' within rate limit, current count: {int(count) + 1}")
        return False
    
    logging.warning(f"User '{user}' exceeded rate limit, current count: {count}")
    return True