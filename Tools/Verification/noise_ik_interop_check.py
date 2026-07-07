"""Mirror of Sources/NMP/NoiseIK.swift logic, verified against the reference
noiseprotocol library for Noise_IK_25519_AESGCM_SHA256."""
import hashlib, hmac as hmac_mod, os
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives import serialization

def sha256(b): return hashlib.sha256(b).digest()
def hmac(k, d): return hmac_mod.new(k, d, hashlib.sha256).digest()
def hkdf2(ck, ikm):
    t = hmac(ck, ikm); o1 = hmac(t, b"\x01"); o2 = hmac(t, o1 + b"\x02"); return o1, o2
def gcm_nonce(n): return b"\x00"*4 + n.to_bytes(8, "big")
def pub(priv): return priv.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
def dh(priv, pub_bytes): return priv.exchange(X25519PublicKey.from_public_bytes(pub_bytes))

class Sym:  # mirrors NoiseSymmetricState + NoiseCipherState
    def __init__(self, name):
        nd = name.encode()
        self.h = nd + b"\x00"*(32-len(nd)) if len(nd) <= 32 else sha256(nd)
        self.ck = self.h; self.k = None; self.n = 0
    def mix_hash(self, d): self.h = sha256(self.h + d)
    def mix_key(self, ikm): self.ck, self.k = hkdf2(self.ck, ikm); self.n = 0
    def enc_hash(self, pt):
        ct = AESGCM(self.k).encrypt(gcm_nonce(self.n), pt, self.h); self.n += 1
        self.mix_hash(ct); return ct
    def dec_hash(self, ct):
        pt = AESGCM(self.k).decrypt(gcm_nonce(self.n), ct, self.h); self.n += 1
        self.mix_hash(ct); return pt
    def split(self): return hkdf2(self.ck, b"")

def initiator_msg1(sym, s_i, e_i, rs_pub, payload):
    sym.mix_hash(rs_pub)                 # pre-message <- s
    out = pub(e_i); sym.mix_hash(pub(e_i))          # e
    sym.mix_key(dh(e_i, rs_pub))                    # es
    out += sym.enc_hash(pub(s_i))                   # s
    sym.mix_key(dh(s_i, rs_pub))                    # ss
    out += sym.enc_hash(payload)
    return out

def initiator_read_msg2(sym, s_i, e_i, msg):
    re = msg[:32]; sym.mix_hash(re)                 # e
    sym.mix_key(dh(e_i, re))                        # ee
    sym.mix_key(dh(s_i, re))                        # se
    return sym.dec_hash(msg[32:])

# --- Reference responder: noiseprotocol library ---
from noise.connection import NoiseConnection, Keypair

s_i = X25519PrivateKey.generate()
e_i = X25519PrivateKey.generate()
s_r = X25519PrivateKey.generate()
s_r_priv_raw = s_r.private_bytes(serialization.Encoding.Raw,
    serialization.PrivateFormat.Raw, serialization.NoEncryption())

resp = NoiseConnection.from_name(b"Noise_IK_25519_AESGCM_SHA256")
resp.set_as_responder()
resp.set_keypair_from_private_bytes(Keypair.STATIC, s_r_priv_raw)
resp.start_handshake()

# msg1: our algorithm -> reference responder
sym = Sym("Noise_IK_25519_AESGCM_SHA256")
sym.mix_hash(b"")  # empty prologue
msg1_payload = b"init-caps"
msg1 = initiator_msg1(sym, s_i, e_i, pub(s_r), msg1_payload)
rx1 = bytes(resp.read_message(msg1))
assert rx1 == msg1_payload, "msg1 payload mismatch"
assert len(msg1) == 32 + 48 + len(msg1_payload) + 16, "msg1 size mismatch"

# msg2: reference responder -> our algorithm
msg2_payload = b"resp-caps"
msg2 = bytes(resp.write_message(msg2_payload))
rx2 = initiator_read_msg2(sym, s_i, e_i, msg2)
assert rx2 == msg2_payload, "msg2 payload mismatch"

# transport keys: our split() vs reference transport encryption
k1, k2 = sym.split()   # k1: initiator->responder, k2: responder->initiator
for i in range(3):
    pt = f"i->r packet {i}".encode()
    ct = AESGCM(k1).encrypt(gcm_nonce(i), pt, b"")
    assert bytes(resp.decrypt(ct)) == pt, "i->r transport key mismatch"
    pt2 = f"r->i packet {i}".encode()
    ct2 = bytes(resp.encrypt(pt2))
    assert AESGCM(k2).decrypt(gcm_nonce(i), ct2, b"") == pt2, "r->i transport key mismatch"

# handshake hash agreement
assert sym.h == resp.get_handshake_hash(), "handshake hash mismatch"
print("ALL CHECKS PASSED: msg1/msg2 framing, payload auth, Split() keys, handshake hash")
print("interop-verified against reference noiseprotocol library")
