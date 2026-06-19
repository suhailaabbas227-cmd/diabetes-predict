# ============================================================
#  Supabase Authentication (cloud)
#  Users are stored in Supabase's cloud database — permanently
#  on the web, surviving restarts.
#  (Do NOT name this file 'supabase_auth' — it clashes with the library.)
# ============================================================

import streamlit as st
from supabase import create_client, Client


@st.cache_resource
def get_client() -> Client:
    """Create the Supabase client (credentials from .streamlit/secrets.toml)."""
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)


def _friendly_error(err: Exception) -> str:
    """Turn raw Supabase errors into clean, user-friendly messages."""
    msg = str(err).lower()
    if "already registered" in msg or "already been registered" in msg:
        return "This email is already registered. Please log in instead."
    if "invalid login" in msg or "invalid credentials" in msg:
        return "Incorrect email or password."
    if "email not confirmed" in msg:
        return "Please confirm your email first — check your inbox (and Spam folder)."
    if "password should be at least" in msg or "weak" in msg:
        return "Password must be at least 6 characters long."
    if "unable to validate email" in msg or "invalid format" in msg:
        return "Please enter a valid email address."
    return f"Something went wrong: {err}"


def sign_up(email: str, password: str):
    """Create a new account. Returns (success: bool, message: str)."""
    email = email.strip().lower()
    if not email or not password:
        return False, "Email and password are required."
    if len(password) < 6:
        return False, "Password must be at least 6 characters long."

    try:
        client = get_client()
        res = client.auth.sign_up({"email": email, "password": password})
        if res.user is None:
            return False, "Could not create the account. Please try again."
        return True, ("We've sent a confirmation email to your inbox. "
                      "Please open it and click the confirmation link, "
                      "then return here and log in. "
                      "(Don't see it? Check your Spam / Junk folder.)")
    except Exception as e:
        return False, _friendly_error(e)


def login(email: str, password: str):
    """Verify a login. Returns (success: bool, message: str)."""
    email = email.strip().lower()
    if not email or not password:
        return False, "Email and password are required."

    try:
        client = get_client()
        res = client.auth.sign_in_with_password({"email": email, "password": password})
        if res.user is None:
            return False, "Incorrect email or password."
        return True, "Login successful!"
    except Exception as e:
        return False, _friendly_error(e)
