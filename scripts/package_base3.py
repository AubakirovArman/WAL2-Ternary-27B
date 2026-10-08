"""CPU-only independent copy and checksum manifest for the compact storage artifact."""
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def scrub(value):
    if isinstance(value, str):
        return 'LOCAL_ARTIFACT/'+Path(value).name if value.startswith('/') else value
    if isinstance(value, dict):
        return {k:scrub(v) for k,v in value.items()}
    if isinstance(value, list):
        return [scrub(v) for v in value]
    return value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path)
    p.add_argument('output',type=Path)
    p.add_argument('--manifest',type=Path,default=ROOT/'model/artifact.json')
    a = p.parse_args()
    artifact=json.loads(a.manifest.read_text())
    meta=json.loads((a.source/'manifest.json').read_text())
    assert meta['format']=='vol2-ternary-base3-v1' and meta['group_size']==128
    assert len(meta['matrices'])==402
    expected={i['file'] for i in meta['matrices'].values()}|{'exceptions.safetensors'}
    actual={f.name for f in a.source.glob('*.safetensors')}
    assert expected==actual
    assert sum((a.source/n).stat().st_size for n in expected)==5846382464
    if a.output.exists():
        raise FileExistsError(a.output)
    pending=a.output.with_name(a.output.name+'.writing')
    pending.mkdir(parents=True,exist_ok=False)
    data=pending/'base3';data.mkdir()
    checksums=[]
    for src in sorted(a.source.iterdir()):
        if not src.is_file() or src.is_symlink():
            raise ValueError('Only regular files expected in frozen base3 source')
        dst=data/src.name
        subprocess.run(['cp','--reflink=auto','--',str(src),str(dst)],check=True)
        if src.stat().st_ino==dst.stat().st_ino and src.stat().st_dev==dst.stat().st_dev:
            raise ValueError('Source and publication files must have distinct inodes')
        before=digest(src)
        if digest(dst)!=before:
            raise ValueError('Copy checksum mismatch')
    # Only provenance host paths are sanitized; weight files remain byte-identical.
    (data/'manifest.json').write_text(json.dumps(scrub(meta),ensure_ascii=False,indent=2)+'\n')
    for file in sorted(data.iterdir()):
        checksums.append(f'{digest(file)}  base3/{file.name}')
    (pending/'SHA256SUMS').write_text('\n'.join(checksums)+'\n')
    shutil.copy2(ROOT/'MODEL_CARD.md',pending/'MODEL_CARD.md')
    shutil.copy2(ROOT/'NOTICE',pending/'NOTICE')
    shutil.copy2(ROOT/'third_party/QWEN_LICENSE',pending/'LICENSE')
    shutil.copy2(a.manifest,pending/'artifact.json')
    (pending/'README.md').write_text(artifact['model']+' compact base3 storage package.\n\n'
        '5.846 GB tensors, 1.739 bits per language parameter; metadata/tokenizer about 5.869 GB.\n'
        'This is not the native PQ2 inference format. Reference loading expands weights for BF16 compute.\n'
        'For fast direct inference use the separate 7.253-GB PQ2 GGUF package and pinned Prism runtime.\n'
        'No FP32 QAT master or optimizer state is included.\n')
    pending.rename(a.output)
    print(json.dumps(dict(passed=True,package=str(a.output.resolve()),files=len(checksums),
                          tensor_bytes=5846382464,scope='Every copied file hashed; ternary symbols already audited by exporter'),indent=2))


if __name__=='__main__':
    main()
