"""Prepare/upload the verified Vol2 inference artifacts, without using GPUs."""
import argparse
import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from verify_model import ROOT, verify

VARIANTS = [('e022-fixed128', 'artifact.json'), ('e020-b1024', 'e020-b1024.json')]
FORMAT = 'vol2-hf-publication-v1'


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.writing')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    tmp.replace(path)


def payload_copy(source, destination):
    """Large immutable publication files may share an inode with the release copy.
    Release copies already have independent inodes from training artifacts.
    Uploading reads the files and never rewrites their contents.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if os.path.samefile(source, destination):
            return
        if source.stat().st_size >= 1024*1024:
            raise FileExistsError(f'Unexpected existing payload: {destination}')
        destination.unlink()
    if source.stat().st_size >= 1024*1024:
        try:
            os.link(source, destination)
            return
        except OSError:
            pass
    shutil.copy2(source, destination)


def prepare(output):
    output.mkdir(parents=True, exist_ok=True)
    (output/'artifacts').mkdir(exist_ok=True)
    old_marker = output/'publication.json'
    if old_marker.exists() and json.loads(old_marker.read_text()).get('format') != FORMAT:
        raise ValueError('Destination contains another publication')
    variants = []
    for label, filename in VARIANTS:
        artifact = json.loads((ROOT/'model'/filename).read_text())
        release = ROOT/'releases'/f'{label}-pq2'
        model = release/artifact['filename']
        verify(model, artifact)
        payload_copy(model, output/artifact['filename'])
        compact = ROOT/'releases'/f'{label}-base3'
        for source in sorted((compact/'base3').iterdir()):
            if not source.is_file() or source.is_symlink():
                raise ValueError('Only regular compact artifact files expected')
            payload_copy(source, output/'base3'/label/source.name)
        shutil.copy2(compact/'SHA256SUMS', output/'base3'/label/'PACKAGE_SHA256SUMS')
        # The package checksum list addresses base3/<file> relative to its wrapper.
        (output/'base3'/label/'PACKAGE_SHA256SUMS').write_text(
            (compact/'SHA256SUMS').read_text().replace('  base3/', '  '))
        shutil.copy2(ROOT/'model'/filename, output/'artifacts'/filename)
        variants.append(dict(label=label, **artifact))
    for source, destination in [(ROOT/'third_party/QWEN_LICENSE',output/'LICENSE'),
                                (ROOT/'NOTICE',output/'NOTICE'),
                                (ROOT/'MODEL_CARD.md',output/'README.md')]:
        shutil.copy2(source, destination)
    for name in ['METHOD_RU.md','EXPERIMENTS_RU.md','BENCHMARKS_RU.md','DATA_RU.md','REPRODUCE_RU.md']:
        payload_copy(ROOT/'docs'/name,output/'docs'/name)
    for source in sorted((ROOT/'evidence').rglob('*')):
        if source.is_file():
            payload_copy(source,output/source.relative_to(ROOT))
    card = (output/'README.md').read_text()
    card += '\n## Файлы этого репозитория\n\n'
    card += ('| Вариант | Native GGUF | Диагностика136 | AIME25 |\n'
             '|---|---|---:|---:|\n')
    for v in variants:
        card += (f"| {v['label']} | [{v['filename']}]({v['filename']}) | "
                 f"{v['diagnostic_correct']}/136 | {v['aime25_correct']}/30 |\n")
    card += ('\nКомпактные base3-файлы расположены в `base3/<вариант>/`. '
             'Native GGUF требует закреплённого Prism/llama.cpp: обычная загрузка '
             'через Transformers или универсальный llama.cpp не заявляется. '
             'Base3-loader reference разворачивает значения для BF16 compute. '
             'FP32 master и optimizer checkpoint здесь отсутствуют.\n'
             '\nВ snapshot-документации могли быть указаны ещё не назначенные URL. '
             'Этот README описывает фактический состав загруженного model repository; '
             'ссылки проекта обновляются после проверки публикации.\n')
    (output/'README.md').write_text(card)
    (output/'.gitattributes').write_text(
        '*.gguf filter=lfs diff=lfs merge=lfs -text\n'
        '*.safetensors filter=lfs diff=lfs merge=lfs -text\n')
    payload_files = [f for f in output.rglob('*') if f.is_file() and '.cache' not in f.relative_to(output).parts]
    marker = dict(format=FORMAT, variants=variants,
                  snapshot=json.loads((ROOT/'PUBLICATION_SNAPSHOT.json').read_text()),
                  prepared_utc=datetime.now(timezone.utc).isoformat(),
                  files=len(payload_files)+int(not old_marker.exists()),
                  bytes=sum(f.stat().st_size for f in payload_files))
    write_json(old_marker,marker)
    print(f'Подготовлено: {len(variants)} кандидата; примерно {marker["bytes"]/1e9:.2f} ГБ.', flush=True)
    return marker


def lfs_sha(file):
    lfs = file.lfs
    return lfs.get('sha256') if isinstance(lfs,dict) else getattr(lfs,'sha256',None)


def verify_remote(api,repo_id,marker):
    # Verify both binary artifacts in a fixed remote commit; public anonymous reads
    # confirm that the repo is actually public, not merely visible to this token.
    info = api.repo_info(repo_id,repo_type='model',token=False)
    if info.private:
        raise ValueError('Expected a public model repository')
    paths = [v['filename'] for v in marker['variants']]
    files = {f.path:f for f in api.get_paths_info(repo_id,paths,revision=info.sha,repo_type='model',token=False)}
    for v in marker['variants']:
        file = files.get(v['filename'])
        if file is None or file.size != v['bytes'] or lfs_sha(file) != v['sha256']:
            raise ValueError(f"Remote file does not match release: {v['filename']}")
    expected = [str(p.relative_to(Path(marker['folder']))) for p in Path(marker['folder']).rglob('*')
                if p.is_file() and '.cache' not in p.relative_to(Path(marker['folder'])).parts]
    published = set(api.list_repo_files(repo_id,revision=info.sha,repo_type='model',token=False))
    missing = set(expected)-published
    if missing:
        raise ValueError(f'Upload incomplete: {len(missing)} missing files')
    return dict(state='published',repo_id=repo_id,url=f'https://huggingface.co/{repo_id}',
                revision=info.sha,verified_public=True,verified_native_sha256=True,
                verified_file_inventory=True,completed_utc=datetime.now(timezone.utc).isoformat(),
                variants=marker['variants'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--folder',type=Path,default=ROOT/'releases/huggingface-upload')
    p.add_argument('--repo',help='Defaults to <authenticated username>/WAL2-Ternary-27B')
    p.add_argument('--workers',type=int,default=8)
    a=p.parse_args()
    if not 1 <= a.workers <= 32:
        p.error('workers must be between1 and32')
    if not a.prepare_only:
        from huggingface_hub import HfApi,get_token
        from huggingface_hub.errors import RepositoryNotFoundError
        if not get_token():
            raise SystemExit('Нет входа в Hugging Face. Выполните hf auth login в терминале; токен не передавайте в чат.')
        api=HfApi()
        profile=api.whoami()
        repo_id=a.repo or profile['name']+'/WAL2-Ternary-27B'
        if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+',repo_id):
            p.error('repo must be owner/name')
        if repo_id.split('/')[0] != profile['name']:
            p.error('This launcher creates only a repo owned by the authenticated user')
    marker=prepare(a.folder)
    if a.prepare_only:
        write_json(ROOT/'logs/huggingface-publication.json',dict(state='prepared_not_uploaded',folder=str(a.folder.resolve())))
        return
    try:
        existing=api.repo_info(repo_id,repo_type='model')
    except RepositoryNotFoundError:
        existing=None
    if existing is not None:
        names=set(api.list_repo_files(repo_id,repo_type='model'))
        if names-{'.gitattributes'} and 'publication.json' not in names:
            raise SystemExit('Этот репозиторий содержит другой проект; содержимое не перезаписано.')
        if 'publication.json' in names:
            from huggingface_hub import hf_hub_download
            remote=json.loads(Path(hf_hub_download(repo_id,'publication.json',repo_type='model')).read_text())
            if remote.get('format') != FORMAT:
                raise SystemExit('Маркер репозитория относится к другому проекту.')
        if existing.private:
            raise SystemExit('Существующий репозиторий приватный; выберите новый публичный repo_id.')
    api.create_repo(repo_id,repo_type='model',private=False,exist_ok=True)
    write_json(ROOT/'logs/huggingface-publication.json',dict(state='uploading',repo_id=repo_id,folder=str(a.folder.resolve())))
    print(f'Загрузка модели: https://huggingface.co/{repo_id}',flush=True)
    api.upload_large_folder(repo_id,folder_path=a.folder,repo_type='model',private=False,
                            num_workers=a.workers,ignore_patterns=['.cache/**'],
                            print_report=True,print_report_every=30)
    marker['folder']=str(a.folder.resolve())
    result=verify_remote(api,repo_id,marker)
    write_json(ROOT/'logs/huggingface-publication.json',result)
    print('Публикация подтверждена: '+result['url'],flush=True)


if __name__=='__main__':
    main()
