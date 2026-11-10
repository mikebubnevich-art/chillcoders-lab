def rotate(items, steps):
    if not items:
        return items
    steps = steps % len(items)
    return items[steps:] + items[:steps]

def is_palindrome(text):
    cleaned = ''.join(ch for ch in text if ch.isalnum())
    return cleaned == cleaned[::-1]

def flatten_once(items):
    return [value for value in items]
