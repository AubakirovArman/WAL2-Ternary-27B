"""Read-only verification of the independent source copy; Python stdlib only."""
from pathlib import Path
import hashlib
import json
import zlib

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path('UPSTREAM_SOURCE')
COPY = ROOT / 'models/qwen3.8-27b-fp8-source'

def digest(path):
    sha = hashlib.sha256()
    crc = 0
    with path.open('rb') as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            sha.update(chunk)
            crc = zlib.crc32(chunk, crc)
    return sha.hexdigest(), f'{crc:08x}'

def main():
    expected = {}
    for line in (SOURCE / 'crc32.txt').read_text().splitlines():
        if line.strip():
            crc, name = line.split(maxsplit=1)
            expected[name.strip().lstrip('*')] = crc.lower()
    records = []
    for original in sorted(SOURCE.iterdir()):
        if not original.is_file():
            continue  # Download cache is not part of the model.
        copied = COPY / original.name
        sha, crc = digest(original)
        copy_sha, copy_crc = digest(copied)
        independent = (original.stat().st_dev, original.stat().st_ino) != (copied.stat().st_dev, copied.stat().st_ino)
        ok = sha == copy_sha and independent and (original.name not in expected or expected[original.name] == crc)
        records.append(dict(file=original.name, bytes=original.stat().st_size,
                            sha256=sha, copy_sha256=copy_sha, crc32=crc,
                            expected_crc32=expected.get(original.name), independent_inode=independent, ok=ok))
        print(original.name, 'OK' if ok else 'FAILED', flush=True)
    result = dict(source=str(SOURCE), copy=str(COPY), files=records,
                  all_ok=all(r['ok'] for r in records),
                  excluded='Nested .cache download metadata; one cache file was unreadable.')
    (ROOT / 'reports/source_verification.json').write_text(json.dumps(result, indent=2))
    if not result['all_ok']:
        raise SystemExit('Verification failed')

if __name__ == '__main__':
    main()
