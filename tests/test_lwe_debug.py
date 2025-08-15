import pytest
import numpy as np
from e8leech_project.crypto.lwe import LWECryptosystem

def test_lwe_single_bit_encryption_decryption():
    dimension = 24
    modulus = 257 # A prime modulus
    lwe = LWECryptosystem(dimension, modulus)
    lwe.generate_keypair()

    # Test encrypting and decrypting a single bit '0'
    plaintext_bit_0 = b'\x00'
    ciphertext_0 = lwe.encrypt(plaintext_bit_0)
    decrypted_0 = lwe.decrypt(ciphertext_0)
    assert decrypted_0 == plaintext_bit_0, f"Decryption of 0 failed: Expected {plaintext_bit_0}, Got {decrypted_0}"

    # Test encrypting and decrypting a single bit '1'
    plaintext_bit_1 = b'\x01'
    ciphertext_1 = lwe.encrypt(plaintext_bit_1)
    decrypted_1 = lwe.decrypt(ciphertext_1)
    assert decrypted_1 == plaintext_bit_1, f"Decryption of 1 failed: Expected {plaintext_bit_1}, Got {decrypted_1}"

def test_lwe_simple_message_encryption_decryption():
    dimension = 24
    modulus = 257
    lwe = LWECryptosystem(dimension, modulus)
    lwe.generate_keypair()

    plaintext = b'Hello'
    ciphertext = lwe.encrypt(plaintext)
    decrypted_text = lwe.decrypt(ciphertext)
    assert decrypted_text == plaintext, f"Decryption of '{plaintext}' failed: Expected {plaintext}, Got {decrypted_text}"


