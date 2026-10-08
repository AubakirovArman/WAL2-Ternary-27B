"""Same frozen statistical panel across parent and all four E020 candidates."""
import json
from pathlib import Path
import sample_e019_trit_movement as sample
ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    sample.OUT=ROOT/'reports/e020-trit-movement'
    if not (sample.OUT/'summary.json').exists():
        paths={'e018':ROOT/'runs/e018-fresh25000/candidates/step-25000'}
        paths.update({f'{b}-{s}':ROOT/f'runs/e020-ab-{b.lower()}/candidates/step-{s:04d}' for b in ('A','B') for s in (512,1024)})
        matrices=json.loads((ROOT/'reports/e018-final-aime10/direct-packed/manifest.json').read_text())['matrices']
        sample.main(paths,matrices)
