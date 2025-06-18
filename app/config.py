from dotenv import load_dotenv
import os

load_dotenv()  # Load from .env file


class VarSettings:
    DEBUG = os.getenv("DEBUG", "False") == "True"
    TYHOON_API_KEY = os.getenv("TYHOON_API_KEY")
