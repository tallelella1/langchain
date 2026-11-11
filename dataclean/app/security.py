from datetime import timedelta

import jwt
from pwdlib import PasswordHash

from .config import Settings
from .models import User, utcnow

password_hash = PasswordHash.recommended()
# Equalize password checking for unknown users.
DUMMY_HASH = password_hash.hash("unused-dummy-password")


def create_token(user: User, settings: Settings) -> str:
    now = utcnow()
    return jwt.encode(
        {"sub": user.id, "iat": now, "exp": now + timedelta(minutes=settings.token_minutes), "iss": "dataclean"},
        settings.jwt_secret,
        algorithm="HS256",
    )


def decode_token(token: str, settings: Settings) -> str:
    payload = jwt.decode(
        token, settings.jwt_secret, algorithms=["HS256"], issuer="dataclean",
        options={"require": ["exp", "sub", "iat", "iss"]},
    )
    if not isinstance(payload["sub"], str):
        raise jwt.InvalidTokenError("Invalid subject")
    return payload["sub"]
