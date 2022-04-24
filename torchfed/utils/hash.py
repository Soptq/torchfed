import hashlib


def hex_hash(data: str) -> str:
    sha = hashlib.sha256()
    sha.update(data.encode())
    return sha.hexdigest()
