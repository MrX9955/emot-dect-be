"""
Authentication routes: signup, login, get current user profile.
"""
from fastapi import APIRouter, HTTPException, status, Depends
from datetime import datetime, timezone
from database import get_database
from models.user import UserCreate, UserLogin, UserResponse
from auth.password_handler import hash_password, verify_password
from auth.jwt_handler import create_access_token
from auth.dependencies import get_current_user

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/signup", status_code=status.HTTP_201_CREATED)
async def signup(user_data: UserCreate):
    """
    Register a new user.
    - Validates email uniqueness
    - Hashes password with bcrypt
    - Returns JWT access token
    """
    db = get_database()

    # Check if email already exists
    existing = await db["users"].find_one({"email": user_data.email})
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered",
        )

    # Store user with hashed password
    user_doc = {
        "name": user_data.name,
        "email": user_data.email,
        "hashed_password": hash_password(user_data.password),
        "created_at": datetime.now(timezone.utc),
    }
    result = await db["users"].insert_one(user_doc)

    token = create_access_token({"sub": user_data.email})
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "id": str(result.inserted_id),
            "name": user_data.name,
            "email": user_data.email,
        },
    }


@router.post("/login")
async def login(credentials: UserLogin):
    """
    Authenticate an existing user.
    - Verifies email and password
    - Returns JWT access token
    """
    db = get_database()

    user = await db["users"].find_one({"email": credentials.email})
    if not user or not verify_password(credentials.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    token = create_access_token({"sub": credentials.email})
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "id": str(user["_id"]),
            "name": user["name"],
            "email": user["email"],
        },
    }


@router.get("/me")
async def get_me(current_user: dict = Depends(get_current_user)):
    """Return the currently authenticated user's profile."""
    return current_user
