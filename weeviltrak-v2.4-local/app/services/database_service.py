"""
Database service for managing Redshift connections.
"""
import os
import redshift_connector
import dotenv
dotenv.load_dotenv()
from app.config.weevilltrak_config import WeevillTrakConfig
from app.settings import setup_logger
from typing import Optional
logger = setup_logger()
class DatabaseManager:
    """
    Manages database connections to Redshift.
    Provides context manager support for safe connection handling.
    """
    def __init__(self, config: Optional[WeevillTrakConfig] = None):
        self.config = config
        self._connection = None


    def get_connection(self):
        """
        Get or create a database connection.
        
        Args:
            config: Configuration object
        
        Returns:
            redshift_connector.Connection: Database connection
        """
        if self._connection is None:
            self._connection = redshift_connector.connect(
                database=os.getenv("DATABASE_NAME"),
                host=os.getenv("REDSHIFT_HOST"),
                port=int(os.getenv("REDSHIFT_PORT", 5439)),
                user=os.getenv("REDSHIFT_USER"),
                password=os.getenv("REDSHIFT_PASSWORD"),
            )
            logger.info("Database connection established")
        
        return self._connection

    def close(self):
        """Close the database connection"""
        if self._connection:
            self._connection.close()
            self._connection = None
            logger.info("Database connection closed")


    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        """Context manager exit - ensures connection is closed"""
        self.close()

    def __del__(self):
        """Cleanup on deletion"""
        self.close()
