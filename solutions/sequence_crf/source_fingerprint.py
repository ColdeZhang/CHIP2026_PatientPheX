"""Role-based dependency fingerprints preserve distinct files sharing a basename."""
import hashlib
from pathlib import Path


def source_hashes(roles):
    return {role:hashlib.sha256(Path(path).read_bytes()).hexdigest() for role,path in roles.items()}
