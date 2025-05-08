import redis

LIMIT = 100
WINDOW = 60

client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

def is_rate_limited(user: str) -> bool:
    key = f'{user}:count'
    count = client.get(key)

    if count == None:
        client.set(key, 1, ex=WINDOW)
        return False 
    
    if count < LIMIT:
        client.incr(key)
        return False
    
    return True
    

