from starlette.config import Config


VERSION = "0.0.1"

config = Config(".env")

APP_ID: str = config("APP_ID", cast=str, default=None)
MODEL_ID: str = config("MODEL_ID", cast=str, default=None)