"""CPU-only file integrity and GGUF layout check; no torch/CUDA imports."""
import argparse
import collections
import hashlib
import json
import math
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXED = {0: ('B', 1), 1: ('b', 1), 2: ('H', 2), 3: ('h', 2),
         4: ('I', 4), 5: ('i', 4), 6: ('f', 4), 7: ('?', 1),
         10: ('Q', 8), 11: ('q', 8), 12: ('d', 8)}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


class HeaderReader:
    def __init__(self, f):
        self.f = f
        self.size = f.seek(0, 2)
        f.seek(0)

    def read(self, n):
        if n < 0 or self.f.tell()+n > self.size:
            raise ValueError('Truncated GGUF header')
        data = self.f.read(n)
        if len(data) != n:
            raise ValueError('Truncated GGUF header')
        return data

    def skip(self, n):
        if n < 0 or self.f.tell()+n > self.size:
            raise ValueError('Truncated GGUF header')
        self.f.seek(n, 1)

    def unpack(self, fmt):
        return struct.unpack('<'+fmt, self.read(struct.calcsize('<'+fmt)))[0]

    def string(self, keep=True):
        n = self.unpack('Q')
        if keep:
            if n > 64*1024*1024:
                raise ValueError('Oversized header string')
            return self.read(n).decode('utf-8')
        self.skip(n)

    def value(self, kind, keep=False):
        if kind in FIXED:
            fmt, size = FIXED[kind]
            if keep:
                return self.unpack(fmt)
            self.skip(size)
        elif kind == 8:
            return self.string(keep)
        elif kind == 9:
            subtype, count = self.unpack('I'), self.unpack('Q')
            if count > self.size:
                raise ValueError('Invalid array count')
            if subtype in FIXED:
                self.skip(FIXED[subtype][1]*count)
            elif subtype == 8:
                for _ in range(count):
                    self.string(False)
            else:
                raise ValueError('Unsupported GGUF array type')
        else:
            raise ValueError(f'Unsupported GGUF value type {kind}')


def header(path):
    wanted = {'general.architecture', 'prism.hadamard.block_size',
              'prism.hadamard.sign_mode'}
    with Path(path).open('rb') as f:
        r = HeaderReader(f)
        if r.read(4) != b'GGUF' or r.unpack('I') != 3:
            raise ValueError('Expected GGUF version 3')
        tensors, fields = r.unpack('Q'), r.unpack('Q')
        if tensors > 100000 or fields > 100000:
            raise ValueError('Invalid header counts')
        metadata = {}
        for _ in range(fields):
            key = r.string()
            value = r.value(r.unpack('I'), key in wanted)
            if key in wanted:
                metadata[key] = value
        counts = collections.Counter()
        parameters = collections.Counter()
        names = set()
        for _ in range(tensors):
            name, ndims = r.string(), r.unpack('I')
            if name in names or not 1 <= ndims <= 4:
                raise ValueError('Duplicate tensor or invalid dimensions')
            names.add(name)
            dims = [r.unpack('Q') for _ in range(ndims)]
            if any(d == 0 for d in dims):
                raise ValueError('Zero-sized tensor')
            kind = r.unpack('I')
            offset = r.unpack('Q')
            if offset >= r.size:
                raise ValueError('Invalid tensor offset')
            counts[kind] += 1
            parameters[kind] += math.prod(dims)
        return {'metadata': metadata, 'tensor_types': dict(counts),
                'parameters_by_type': dict(parameters), 'tensor_count': tensors}


def verify_layout(path, manifest):
    result = header(path)
    if result['metadata'] != {'general.architecture': 'qwen35',
                              'prism.hadamard.block_size': 128,
                              'prism.hadamard.sign_mode': 'identity'}:
        raise ValueError('Architecture or rotation differs from release')
    if result['tensor_types'] != {142: 402, 0: 449}:
        raise ValueError('Expected 402 PQ2_0 and 449 F32 tensors')
    if result['parameters_by_type'] != {142: manifest['ternary_parameters'],
                                       0: manifest['exceptions_parameters']}:
        raise ValueError('Parameter counts differ from release')
    return dict(passed=True, scope='GGUF header layout only', **result)


def verify(path, manifest):
    path = Path(path)
    if path.stat().st_size != manifest['bytes']:
        raise ValueError('File size differs from artifact manifest')
    actual = digest(path)
    if actual != manifest['sha256']:
        raise ValueError('SHA256 differs from artifact manifest')
    result = verify_layout(path, manifest)
    result.update(scope='Release size, SHA256 and GGUF header layout',
                  sha256=actual, bytes=path.stat().st_size)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('model', type=Path)
    p.add_argument('--manifest', type=Path, default=ROOT/'model/artifact.json')
    p.add_argument('--layout-only', action='store_true',
                   help='Check native layout without hashing the file; not a release integrity check')
    a = p.parse_args()
    operation = verify_layout if a.layout_only else verify
    print(json.dumps(operation(a.model, json.loads(a.manifest.read_text())),
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
