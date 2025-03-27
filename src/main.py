"""
Main application module.
"""
import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from src.utils.logger import get_logger

# Load environment variables from .env file
load_dotenv()

# Get environment
ENV = os.getenv("ENV", "development")

# Initialize logger
logger = get_logger("main")


def main():
    """
    Main entry point for the application.
    """
    logger.info(f"Application started in {ENV} mode")


if __name__ == "__main__":
    main()
