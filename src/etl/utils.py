# borrowed from https://code.activestate.com/recipes/498104-isbn-13-converter/
def check_digit_13(isbn: str):
    assert len(isbn) == 12
    sum = 0
    for i in range(len(isbn)):
        c = int(isbn[i])
        if i % 2:
            w = 3
        else:
            w = 1
        sum += w * c
    r = 10 - (sum % 10)
    if r == 10:
        return "0"
    else:
        return str(r)


# borrowed from https://code.activestate.com/recipes/498104-isbn-13-converter/
def convert_10_to_13(isbn: str):
    prefix = "978" + isbn[:-1]
    check = check_digit_13(prefix)
    return prefix + check


def convert_to_13(isbn: str):
    if len(isbn) == 13:
        return isbn
    if len(isbn) == 10 and isbn[:-1].isdigit():
        return convert_10_to_13(isbn)
    return isbn
