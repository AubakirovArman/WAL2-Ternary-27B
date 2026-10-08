"""Build a separate verified inference package; do not alter experiment weights."""
import argparse
import json
import shutil
import subprocess
from pathlib import Path
from verify_model import ROOT, verify


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('model', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--manifest', type=Path, default=ROOT/'model/artifact.json')
    a = p.parse_args()
    manifest = json.loads(a.manifest.read_text())
    if a.output.exists():
        raise FileExistsError(a.output)
    verify(a.model, manifest)
    pending = a.output.with_name(a.output.name+'.writing')
    pending.mkdir(parents=True, exist_ok=False)
    destination = pending/manifest['filename']
    subprocess.run(['cp', '--reflink=auto', '--', str(a.model), str(destination)], check=True)
    if a.model.stat().st_ino == destination.stat().st_ino and a.model.stat().st_dev == destination.stat().st_dev:
        raise ValueError('Package must not share inode with experiment weights')
    verify(destination, manifest)
    for src, name in [(ROOT/'MODEL_CARD.md','MODEL_CARD.md'), (a.manifest,'artifact.json'),
                      (ROOT/'third_party/QWEN_LICENSE','LICENSE'),(ROOT/'NOTICE','NOTICE')]:
        shutil.copy2(src,pending/name)
    (pending/'SHA256SUMS').write_text(f"{manifest['sha256']}  {manifest['filename']}\n")
    (pending/'README.md').write_text(manifest['model']+' — native PQ2 inference package.\n\n'
        '7,253 GB / 6,755 GiB; 2.125 bits per ternary weight including group scale; '
        '2.157 bits per language parameter for entire file.\n'
        'Requires PrismML-Eng/llama.cpp commit '+manifest['runtime_commit']+'.\n'
        'Not a resumable FP32 QAT master. Source project contains runtime build/serve instructions.\n'
        f"Benchmark of this checkpoint: {manifest['diagnostic_correct']}/136 development tasks, "
        f"AIME25 {manifest['aime25_correct']}/30.\n"
        'Full GSM8K84.69% belongs to an earlier E017 checkpoint, not this file.\n')
    pending.rename(a.output)
    print(json.dumps(dict(passed=True, package=str(a.output.resolve()), model_bytes=manifest['bytes']),indent=2))


if __name__ == '__main__':
    main()
