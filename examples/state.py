import json
from promptic import State, llm


class RedisState(State):
    def __init__(self, redis_client):
        super().__init__()
        self.redis = redis_client
        self.key = "chat_history"

    def add_message(self, message):
        self.redis.rpush(self.key, json.dumps(message))

    def get_messages(self, limit=None):
        messages = self.redis.lrange(self.key, 0, -1)
        return [json.loads(m) for m in messages][-limit:] if limit else messages

    def clear(self):
        self.redis.delete(self.key)


@llm(state=RedisState(redis_client))
def persistent_chat(message):
    """Chat: {message}"""
