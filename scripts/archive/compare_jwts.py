"""Compare our JWT implementation with the official SDK."""
import secrets
import time
import jwt as pyjwt
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from coinbase import jwt_generator
from libs.config.config import Settings

# Load credentials
config = Settings()
api_key = config.effective_api_key
private_key_pem = config.effective_api_secret.replace('\\n', '\n')

print("=" * 70)
print("JWT COMPARISON")
print("=" * 70)
print()

# Official SDK JWT
request_path = "/api/v3/brokerage/products/BTC-USD/candles"
official_jwt = jwt_generator.build_rest_jwt(
    f"GET {request_path}",
    api_key,
    private_key_pem
)
print("Official SDK JWT (first 100 chars):")
print(official_jwt[:100] + "...")
print()

# Decode to see payload
import json
decoded_official = pyjwt.decode(official_jwt, options={"verify_signature": False})
print("Official JWT Payload:")
print(json.dumps(decoded_official, indent=2))
print()
print("Official JWT Header:")
header_official = pyjwt.get_unverified_header(official_jwt)
print(json.dumps(header_official, indent=2))
print()

# Our implementation
private_key = serialization.load_pem_private_key(
    private_key_pem.encode('utf-8'),
    password=None,
    backend=default_backend()
)

jwt_uri = f"GET api.coinbase.com{request_path}"
payload = {
    'sub': api_key,
    'iss': 'cdp',
    'nbf': int(time.time()),
    'exp': int(time.time()) + 120,
    'uri': jwt_uri
}
headers = {
    'kid': api_key,
    'nonce': secrets.token_hex()
}
our_jwt = pyjwt.encode(payload, private_key, algorithm='ES256', headers=headers)

print("Our JWT (first 100 chars):")
print(our_jwt[:100] + "...")
print()

decoded_ours = pyjwt.decode(our_jwt, options={"verify_signature": False})
print("Our JWT Payload:")
print(json.dumps(decoded_ours, indent=2))
print()
print("Our JWT Header:")
header_ours = pyjwt.get_unverified_header(our_jwt)
print(json.dumps(header_ours, indent=2))
print()

# Find differences
print("=" * 70)
print("DIFFERENCES:")
print("=" * 70)
for key in set(list(decoded_official.keys()) + list(decoded_ours.keys())):
    official_val = decoded_official.get(key, "MISSING")
    our_val = decoded_ours.get(key, "MISSING")
    if official_val != our_val:
        print(f"  {key}:")
        print(f"    Official: {official_val}")
        print(f"    Ours:     {our_val}")
