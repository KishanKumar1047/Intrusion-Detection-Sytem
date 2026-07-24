"""
face_db.py
----------

Helper functions to load/save the registered-face encodings database.
The database is stored as a pickle file at data/encodings.pkl, in the format:

    { "person_name": [encoding1, encoding2, ...], ... }

Storing multiple samples per person (different angles / lighting) improves
recognition accuracy.
"""

import pickle
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "encodings.pkl")


def load_database():
    """Loads the existing face database. Returns an empty dict if none exists yet."""
    if not os.path.exists(DB_PATH):
        return {}
    with open(DB_PATH, "rb") as f:
        return pickle.load(f)


def save_database(db):
    """Saves the database to disk as a pickle file."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with open(DB_PATH, "wb") as f:
        pickle.dump(db, f)


def list_registered_names():
    """Returns a list of all registered person names."""
    return list(load_database().keys())
