from dotenv import load_dotenv
import os

load_dotenv()

TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")