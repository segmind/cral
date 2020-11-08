import xxhash

BUFF_SIZE = 65536


def hashFile(file_path):
    with open(file_path, encoding='Latin-1') as file:
        hs = xxhash.xxh64()
        while True:
            data = file.read(BUFF_SIZE)
            if not data:
                break
            data = data.encode()
            hs.update(data)
        return hs.hexdigest()
    return None


def hashStr(string):
    return xxhash.xxh64(string.encode()).hexdigest()
