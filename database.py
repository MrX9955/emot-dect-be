"""
MongoDB async connection using Motor.
Provides a shared database client across the application.
"""
from motor.motor_asyncio import AsyncIOMotorClient
from config import get_settings

settings = get_settings()

client: AsyncIOMotorClient = None


async def connect_db():
    """Initialize the MongoDB connection."""
    global client
    client = AsyncIOMotorClient(settings.mongodb_url)
    print(f"✅ Connected to MongoDB: {settings.mongodb_url}")


async def close_db():
    """Close the MongoDB connection."""
    global client
    if client:
        client.close()
        print("🔌 MongoDB connection closed.")


def get_database():
    """Return the active database instance."""
    return client[settings.database_name]
