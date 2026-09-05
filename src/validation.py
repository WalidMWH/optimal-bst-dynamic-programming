"""Input parsing and validation for the optimal binary search tree program.

Everything here runs before the DP: the two padded lists must match in length,
the keys must be strictly increasing, and the probabilities must be non-negative
and sum to 1 within a small tolerance. Input comes from a file, an interactive
prompt or a seeded random generator, and any fault raises InvalidInputError.
"""

import random

DEFAULT_TOLERANCE = 1e-6


class InvalidInputError(ValueError):
    """Raised for any malformed or inconsistent problem input."""


def validate(
    keys: list[int],
    probabilities: list[float],
    tolerance: float = DEFAULT_TOLERANCE,
) -> None:
    """Raise InvalidInputError naming the first fault in the padded input lists."""
    if len(keys) != len(probabilities):
        raise InvalidInputError(
            f"key list and probability list differ in length: "
            f"{len(keys)} against {len(probabilities)}"
        )

    n = len(keys) - 1  # index 0 is padding, not a key
    if n < 1:
        raise InvalidInputError(f"at least one key is required, got n = {n}")

    for i in range(1, n):
        if keys[i] == keys[i + 1]:
            raise InvalidInputError(
                f"keys must be distinct: key {i} and key {i + 1} are both {keys[i]}"
            )
        if keys[i] > keys[i + 1]:
            raise InvalidInputError(
                f"keys must increase: key {i} = {keys[i]} comes before "
                f"key {i + 1} = {keys[i + 1]}"
            )

    for i in range(1, n + 1):
        if probabilities[i] < 0:
            raise InvalidInputError(
                f"probability of key {keys[i]} is negative: {probabilities[i]}"
            )

    total = sum(probabilities[1:])
    if abs(total - 1.0) > tolerance:
        raise InvalidInputError(
            f"probabilities must sum to 1, they sum to {total:.10g} "
            f"(tolerance {tolerance:g})"
        )


def parse_lines(lines: list[str]) -> tuple[list[int], list[float]]:
    """Parse the text input format into padded lists and validate them."""
    # Original line numbers are kept alongside the text so errors can point at
    # the offending line even though blanks and comments are dropped here.
    numbered = [
        (number, line.strip())
        for number, line in enumerate(lines, start=1)
        if line.strip() and not line.strip().startswith("#")
    ]

    if not numbered:
        raise InvalidInputError(
            "input is empty: the first line must hold the number of keys"
        )

    count_line, count_text = numbered[0]
    try:
        n = int(count_text)
    except ValueError:
        raise InvalidInputError(
            f"line {count_line}: number of keys is not an integer: {count_text!r}"
        ) from None

    if n < 1:
        raise InvalidInputError(
            f"line {count_line}: number of keys must be at least 1, got {n}"
        )

    data = numbered[1:]
    if len(data) < n:
        raise InvalidInputError(
            f"expected {n} key lines after line {count_line}, found {len(data)}"
        )

    keys: list[int] = [0]
    probabilities: list[float] = [0.0]
    for number, text in data[:n]:
        fields = text.split()
        if len(fields) != 2:
            raise InvalidInputError(
                f"line {number}: expected a key and a probability, "
                f"found {len(fields)} field(s): {text!r}"
            )

        try:
            key = int(fields[0])
        except ValueError:
            raise InvalidInputError(
                f"line {number}: key is not an integer: {fields[0]!r}"
            ) from None

        try:
            probability = float(fields[1])
        except ValueError:
            raise InvalidInputError(
                f"line {number}: probability is not a number: {fields[1]!r}"
            ) from None

        keys.append(key)
        probabilities.append(probability)

    validate(keys, probabilities)
    return keys, probabilities


def load_from_file(path: str) -> tuple[list[int], list[float]]:
    """Read an input file and parse it, reporting read failures as input errors."""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            lines = handle.readlines()
    except OSError as error:
        # A missing or unreadable file is the user's input problem, not a crash.
        raise InvalidInputError(f"cannot read input file {path!r}: {error}") from None

    return parse_lines(lines)


def _prompt(message: str) -> str:
    try:
        return input(message)
    except EOFError:
        raise InvalidInputError("input ended before all keys were entered") from None


def read_interactive() -> tuple[list[int], list[float]]:
    """Prompt for the keys and probabilities on stdin, reprompting on a bad entry."""
    while True:
        text = _prompt("number of keys: ").strip()
        try:
            n = int(text)
        except ValueError:
            print("  not an integer, try again")
            continue
        if n < 1:
            print("  at least one key is needed, try again")
            continue
        break

    keys: list[int] = [0]
    probabilities: list[float] = [0.0]
    for i in range(1, n + 1):
        while True:
            fields = _prompt(f"key {i} and its probability: ").split()
            if len(fields) != 2:
                print("  enter two values separated by a space")
                continue
            try:
                key = int(fields[0])
                probability = float(fields[1])
            except ValueError:
                print("  the key must be an integer and the probability a number")
                continue
            keys.append(key)
            probabilities.append(probability)
            break

    # Ordering and the probability sum can only be judged once everything is in,
    # so those faults are raised rather than reprompted.
    validate(keys, probabilities)
    return keys, probabilities


def generate_random(n: int, seed: int | None = None) -> tuple[list[int], list[float]]:
    """Generate n evenly spaced keys with random probabilities summing to 1."""
    if n < 1:
        raise InvalidInputError(f"at least one key is required, got n = {n}")

    generator = random.Random(seed)
    # random() can return exactly 0.0; the shift keeps every key reachable with
    # a non-zero probability.
    epsilon = 1e-9
    weights = [generator.random() + epsilon for _ in range(n)]
    total = sum(weights)

    keys = [0] + [10 * (i + 1) for i in range(n)]
    probabilities = [0.0] + [weight / total for weight in weights]

    validate(keys, probabilities)
    return keys, probabilities