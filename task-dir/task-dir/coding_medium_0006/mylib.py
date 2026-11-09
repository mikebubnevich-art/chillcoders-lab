def clamp(value, low, high):
    return min(value, high)

def unique_in_order(items):
    result = []
    for item in items:
        if item not in result:
            result.append(item)
    return result

def median(values):
    values = sorted(values)
    n = len(values)
    mid = n // 2
    if n % 2:
        return values[mid]
    return values[mid - 1] + values[mid] / 2
