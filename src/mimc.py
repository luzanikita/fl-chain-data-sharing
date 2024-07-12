# Reference: https://github.com/arnaucube/mimc-rs/blob/master/src/lib.rs

from Crypto.Hash import keccak

PRIME = 21888242871839275222246405745257275088548364400416034343698204186575808495617
SEED = "mimc"


class Constants:
    def __init__(self, n_rounds, cts):
        self.n_rounds = n_rounds
        self.cts = cts


class Fr:
    def __init__(self, value):
        self.value = value % PRIME

    def __add__(self, other):
        return Fr((self.value + other.value) % PRIME)

    def __mul__(self, other):
        return Fr((self.value * other.value) % PRIME)

    def __int__(self):
        return self.value


class Mimc7:
    def __init__(self, n_rounds):
        self.constants = generate_constants(n_rounds)

    def hash(self, x_in, k):
        h = Fr(0)
        for i in range(self.constants.n_rounds):
            if i == 0:
                t = Fr(x_in.value + k.value)
            else:
                t = Fr(h.value + k.value + self.constants.cts[i].value)
            t2 = t * t
            t4 = t2 * t2
            t6 = t4 * t2
            t7 = t6 * t
            h = t7
        return Fr(h.value + k.value)

    def multi_hash(self, arr, key):
        r = key
        for x in arr:
            h = self.hash(x, r)
            r += x + h
        return r


def generate_constants(n_rounds):
    cts = get_constants(SEED, n_rounds)
    return Constants(n_rounds, cts)


def get_constants(seed, n_rounds):
    cts = [Fr(0)]
    h = keccak.new(digest_bits=256).update(seed.encode()).digest()

    c = int.from_bytes(h, "big")
    for _ in range(1, n_rounds):
        h = keccak.new(digest_bits=256).update(c.to_bytes(32, "big")).digest()
        c = int.from_bytes(h, "big")
        n = c % PRIME
        cts.append(Fr(n))
    return cts


def test_generate_constants():
    constants = generate_constants(91)
    expected = 0x2E2EBBB178296B63D88EC198F0976AD98BC1D4EB0D921DDD2EB86CB7E70A98E5
    assert int(constants.cts[1]) == expected, f"Expected {expected}, got {int(constants.cts[1])}"
    print("test_generate_constants passed")


def test_mimc():
    b1, b2, b3 = Fr(1), Fr(2), Fr(3)
    mimc7 = Mimc7(91)

    h1 = mimc7.hash(b1, b2)
    expected = 0x176C6EEFC3FDF8D6136002D8E6F7A885BBD1C4E3957B93DDC1EC3AE7859F1A08
    assert int(h1) == expected, f"Expected {expected:x}, got {int(h1):x}"

    h1 = mimc7.multi_hash([b1, b2, b3], Fr(0))
    expected = 0x25F5A6429A9764564BE3955E6F56B0B9143C571528FD30A80AE6C27DC8B4A40C
    assert int(h1) == expected, f"Expected {expected:x}, got {int(h1):x}"

    b12, b45, b78, b41 = Fr(12), Fr(45), Fr(78), Fr(41)

    h1 = mimc7.multi_hash([b12], Fr(0))
    expected = 0x237C92644DBDDB86D8A259E0E923AAAB65A93F1EC5758B8799988894AC0958FD
    assert int(h1) == expected, f"Expected {expected:x}, got {int(h1):x}"

    mh2 = mimc7.hash(b12, b45)
    expected = 0x2BA7EBAD3C6B6F5A20BDECBA2333C63173CA1A5F2F49D958081D9FA7179C44E4
    assert int(mh2) == expected, f"Expected {expected:x}, got {int(mh2):x}"

    h2 = mimc7.multi_hash([b78, b41], Fr(0))
    expected = 0x067F3202335EA256AE6E6AADCD2D5F7F4B06A00B2D1E0DE903980D5AB552DC70
    assert int(h2) == expected, f"Expected {expected:x}, got {int(h2):x}"

    h1 = mimc7.multi_hash([b12, b45], Fr(0))
    expected = 0x15FF7FE9793346A17C3150804BCB36D161C8662B110C50F55CCB7113948D8879
    assert int(h1) == expected, f"Expected {expected:x}, got {int(h1):x}"

    h1 = mimc7.multi_hash([b12, b45, b78, b41], Fr(0))
    expected = 0x284BC1F34F335933A23A433B6FF3EE179D682CD5E5E2FCDD2D964AFA85104BEB
    assert int(h1) == expected, f"Expected {expected:x}, got {int(h1):x}"

    r_1 = Fr(21888242871839275222246405745257275088548364400416034343698204186575808495616)
    h1 = mimc7.multi_hash([r_1], Fr(0))
    expected = 0x0A4FFFE99225F9972EC39FD780DD084F349286C723D4DD42AD05E2E7421FEF0E
    assert int(h1) == expected, f"Expected {expected:x}, got {int(h1):x}"

    print("test_mimc passed")


def hash_model_weights(weights, n_rounds=91):
    flat_weights = [w for layer in weights for w in layer.flatten()]
    fr_weights = [Fr(int(w * 1e6)) for w in flat_weights]  # Multiply by 1e6 to preserve some decimal places
    mimc = Mimc7(n_rounds)
    result = mimc.multi_hash(fr_weights, Fr(0))

    return int(result)


if __name__ == "__main__":
    test_generate_constants()
    test_mimc()
    print("All tests passed!")
