import base64
import json
import os
from typing import Any

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from flask import current_app
from sqlalchemy.orm.attributes import set_committed_value

DEK_WRAP_AAD = b"coachmim:dek-wrap:v1"
DATA_BLOB_VERSION = 1
DEK_BLOB_VERSION = 1


class EncryptionConfigError(RuntimeError):
    pass


def _normalize_key_bytes(raw: str | bytes | None) -> bytes | None:
    if raw is None:
        return None
    if isinstance(raw, bytes):
        return raw if len(raw) == 32 else None

    text = raw.strip()
    if not text:
        return None

    if len(text) == 64:
        try:
            decoded = bytes.fromhex(text)
            if len(decoded) == 32:
                return decoded
        except ValueError:
            pass

    padded = text + ("=" * ((4 - len(text) % 4) % 4))
    decode_attempts = []
    try:
        decode_attempts.append(base64.urlsafe_b64decode(padded))
    except Exception:
        pass
    try:
        decode_attempts.append(base64.b64decode(padded))
    except Exception:
        pass

    for candidate in decode_attempts:
        if len(candidate) == 32:
            return candidate
    return None


def validate_encryption_configuration(raw_key: str | bytes | None, required: bool) -> None:
    if required and _normalize_key_bytes(raw_key) is None:
        raise EncryptionConfigError(
            "ENCRYPTION_MASTER_KEY must be set to a 32-byte key (base64/urlsafe-base64/hex)."
        )


def _master_key_or_none() -> bytes | None:
    raw = current_app.config.get("ENCRYPTION_MASTER_KEY")
    return _normalize_key_bytes(raw)


def encryption_enabled() -> bool:
    return _master_key_or_none() is not None


def _ensure_master_key() -> bytes | None:
    key = _master_key_or_none()
    if key:
        return key

    if current_app.config.get("ENCRYPTION_REQUIRED"):
        raise EncryptionConfigError(
            "ENCRYPTION_MASTER_KEY is required but not configured."
        )
    return None


def _wrap_dek(master_key: bytes, dek: bytes) -> bytes:
    nonce = os.urandom(12)
    ciphertext = AESGCM(master_key).encrypt(nonce, dek, DEK_WRAP_AAD)
    return bytes([DEK_BLOB_VERSION]) + nonce + ciphertext


def _unwrap_dek(master_key: bytes, wrapped: bytes) -> bytes:
    if not wrapped or len(wrapped) < 14:
        raise EncryptionConfigError("Wrapped DEK blob is invalid.")
    version = wrapped[0]
    if version != DEK_BLOB_VERSION:
        raise EncryptionConfigError("Unsupported wrapped DEK version.")
    nonce = wrapped[1:13]
    ciphertext = wrapped[13:]
    return AESGCM(master_key).decrypt(nonce, ciphertext, DEK_WRAP_AAD)


def ensure_user_dek(user, *, create_if_missing: bool = True) -> bytes | None:
    master_key = _ensure_master_key()
    if master_key is None:
        return None

    wrapped = getattr(user, "encrypted_dek", None)
    if wrapped:
        return _unwrap_dek(master_key, wrapped)

    if not create_if_missing:
        return None

    dek = os.urandom(32)
    user.encrypted_dek = _wrap_dek(master_key, dek)
    return dek


def _to_json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _from_json_bytes(raw: bytes) -> dict[str, Any]:
    data = json.loads(raw.decode("utf-8"))
    return data if isinstance(data, dict) else {}


def _data_aad(scope: str, user_id: int) -> bytes:
    return f"coachmim:{scope}:uid:{user_id}:v1".encode("utf-8")


def encrypt_payload_for_user(user, payload: dict[str, Any], *, scope: str) -> bytes | None:
    if not payload:
        return None

    dek = ensure_user_dek(user, create_if_missing=True)
    if dek is None:
        return None

    nonce = os.urandom(12)
    aad = _data_aad(scope, user.id)
    ciphertext = AESGCM(dek).encrypt(nonce, _to_json_bytes(payload), aad)
    return bytes([DATA_BLOB_VERSION]) + nonce + ciphertext


def decrypt_payload_for_user(user, encrypted_blob: bytes | None, *, scope: str) -> dict[str, Any]:
    if not encrypted_blob:
        return {}

    dek = ensure_user_dek(user, create_if_missing=False)
    if dek is None:
        return {}

    if len(encrypted_blob) < 14:
        return {}
    version = encrypted_blob[0]
    if version != DATA_BLOB_VERSION:
        return {}
    nonce = encrypted_blob[1:13]
    ciphertext = encrypted_blob[13:]

    try:
        plaintext = AESGCM(dek).decrypt(nonce, ciphertext, _data_aad(scope, user.id))
        return _from_json_bytes(plaintext)
    except Exception:
        current_app.logger.warning("Failed to decrypt %s payload for user_id=%s", scope, user.id)
        return {}


def _is_blank(value: Any) -> bool:
    return value in (None, "", [], {})


def encrypt_model_fields(
    *,
    user,
    model,
    encrypted_attr: str,
    fields: list[str],
    scope: str,
) -> None:
    if not fields:
        return

    payload = {}
    for field in fields:
        value = getattr(model, field, None)
        if _is_blank(value):
            continue
        payload[field] = value

    encrypted = encrypt_payload_for_user(user, payload, scope=scope) if payload else None
    setattr(model, encrypted_attr, encrypted)

    if encrypted is not None:
        for field in fields:
            setattr(model, field, None)


def hydrate_model_fields(
    *,
    user,
    model,
    encrypted_attr: str,
    fields: list[str],
    scope: str,
) -> None:
    encrypted_blob = getattr(model, encrypted_attr, None)
    if not encrypted_blob:
        return

    payload = decrypt_payload_for_user(user, encrypted_blob, scope=scope)
    if not payload:
        return

    for field in fields:
        if field in payload:
            set_committed_value(model, field, payload[field])
