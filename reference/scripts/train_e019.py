"""E019 continuation from the saved E018 final step, without relabeling its run."""
import hashlib
import json
import sys
from pathlib import Path

import train_reasoning_qat as qat

ROOT = Path(__file__).resolve().parents[1]
PARENT_RUN = ROOT / 'runs/e018-fresh25000'
PACKED = ROOT / 'reports/e018-final-aime10/direct-packed'
STANDARD_RESOLVE = qat.resolve_parent


def resolve_e019_parent(run, candidate='selected'):
    if run.resolve() != PARENT_RUN.resolve():
        return STANDARD_RESOLVE(run, candidate)
    if candidate != 'selected':
        raise ValueError('E019 continuation requires the selected E018 final checkpoint')
    raw = (run / 'metrics.json').read_bytes()
    metrics = json.loads(raw)
    config = json.loads((run / 'config.json').read_text())
    skipped = json.loads((run / 'finalization-skipped.json').read_text())
    latent = (run / skipped['checkpoint']).resolve()
    if latent != (run / 'candidates/step-25000').resolve():
        raise ValueError('Unexpected interrupted-finalization checkpoint')
    if latent != (ROOT / metrics['selected_latent']).resolve():
        raise ValueError('Interrupted checkpoint is not the selected checkpoint')
    check = metrics['checks'][-1]
    checkpoint_metadata = json.loads((latent / 'metadata.json').read_text())
    if check != checkpoint_metadata or check['step'] != config['steps'] or not check['eligible']:
        raise ValueError('Final checkpoint lacks its completed eligible validation')
    last = json.loads((run / 'training.jsonl').read_text().splitlines()[-1])
    if last['step'] != config['steps']:
        raise ValueError('Parent optimizer updates are incomplete')
    manifest = json.loads((latent / 'manifest.json').read_text())
    for filename in set(manifest.values()) | {'training_state.pt'}:
        if not (latent / filename).is_file():
            raise ValueError('Missing parent checkpoint file: ' + filename)
    packed_manifest = json.loads((PACKED / 'manifest.json').read_text())
    latent_sha = hashlib.sha256((latent / 'manifest.json').read_bytes()).hexdigest()
    if Path(packed_manifest['latent_source']).resolve() != latent or packed_manifest['latent_manifest_sha256'] != latent_sha:
        raise ValueError('Packed parent and latent parent differ')
    export = json.loads((ROOT / 'reports/e018-final-aime10/e018-final.export.json').read_text())
    packed_sha = hashlib.sha256((PACKED / 'manifest.json').read_bytes()).hexdigest()
    if export['checkpoint_manifest_sha256'] != packed_sha or not export['all_trits_roundtrip_exact']:
        raise ValueError('Packed export provenance check failed')
    metrics = dict(metrics, selected_latent=str(latent), selected_packed=str(PACKED),
                   selected_validation=check['old'], selected_reasoning=check['reasoning'])
    descriptor = dict(latent=str(latent), packed=str(PACKED), step=check['step'],
                      original_run_state=metrics['state'], finalization_skipped=skipped,
                      optimizer_updates_completed=True, saved_validation=check,
                      latent_manifest_sha256=latent_sha, packed_manifest_sha256=packed_sha,
                      original_metrics_sha256=hashlib.sha256(raw).hexdigest(),
                      load_identity_gate='Full original 128-example CE must match saved checkpoint CE before updates')
    (ROOT / 'reports/e019-training-parent.json').write_text(json.dumps(descriptor, indent=2))
    return latent, PACKED, metrics, hashlib.sha256(raw).hexdigest()


if __name__ == '__main__':
    if '--check-parent' in sys.argv:
        parent, packed, metrics, digest = resolve_e019_parent(PARENT_RUN)
        print(json.dumps(dict(parent=str(parent), packed=str(packed),
                              original_state=metrics['state'], expected_old_ce=metrics['selected_validation']['answer_ce'],
                              provenance_passed=True), indent=2))
    else:
        qat.resolve_parent = resolve_e019_parent
        qat.main()
