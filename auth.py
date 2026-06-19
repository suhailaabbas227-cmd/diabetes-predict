# ============================================================
#  Simple Authentication Helper (login / sign-up)
#  Users ko users.json mein store karta hai.
#  Passwords plain nahi — salt + SHA-256 hash ban ke save hote hain.
# ============================================================

import json
import hashlib
import secrets
from pathlib import Path

USERS_FILE = Path("users.json")


def _load_users() -> dict:
    """users.json padho. Khaali ya missing ho to {} return karo."""
    if not USERS_FILE.exists() or USERS_FILE.stat().st_size == 0:
        return {}
    try:
        return json.loads(USERS_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _save_users(users: dict) -> None:
    USERS_FILE.write_text(json.dumps(users, indent=2), encoding="utf-8")


def _hash_password(password: str, salt: str) -> str:
    """salt + password ko SHA-256 se hash karo."""
    return hashlib.sha256((salt + password).encode("utf-8")).hexdigest()


def sign_up(username: str, email: str, password: str):
    """Naya account banao. Returns (success: bool, message: str)."""
    username = username.strip().lower()
    email = email.strip()

    if not username or not password:
        return False, "Username aur password zaroori hai."
    if len(username) < 3:
        return False, "Username kam se kam 3 characters ka ho."
    if len(password) < 6:
        return False, "Password kam se kam 6 characters ka ho."

    users = _load_users()
    if username in users:
        return False, "Ye username pehle se maujood hai. Doosra try karein."

    salt = secrets.token_hex(16)
    users[username] = {
        "email": email,
        "salt": salt,
        "password_hash": _hash_password(password, salt),
    }
    _save_users(users)
    return True, "Account ban gaya! Ab Login tab se login karein."


def login(username: str, password: str):
    """Login verify karo. Returns (success: bool, message: str)."""
    username = username.strip().lower()
    users = _load_users()

    user = users.get(username)
    if user is None:
        return False, "Ye username register nahi hai."
    if _hash_password(password, user["salt"]) != user["password_hash"]:
        return False, "Password galat hai."

    return True, "Login successful!"
