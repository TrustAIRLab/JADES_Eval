"""Offline release checks: identity, version, archive boundaries and checksums."""
import ast
import hashlib
import json
import os
from pathlib import Path
import tarfile
import tomllib
import zipfile

root = Path(__file__).resolve().parents[1]
project = tomllib.loads((root / 'pyproject.toml').read_text())['project']
version = project['version']
assert project['name'] == 'jades-eval'
node = ast.parse((root / 'src/jades/__init__.py').read_text())
code_version = next(ast.literal_eval(n.value) for n in node.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == '__version__' for t in n.targets))
assert code_version == version
if os.environ.get('GITHUB_REF', '').startswith('refs/tags/'):
    assert os.environ['GITHUB_REF_NAME'] == 'v' + version, 'Tag must match package version'

files = sorted((root / 'dist').iterdir())
expected = {f'jades_eval-{version}-py3-none-any.whl', f'jades_eval-{version}.tar.gz'}
assert {p.name for p in files} == expected, 'Only current wheel and sdist may be published'
for archive in files:
    if archive.suffix == '.whl':
        with zipfile.ZipFile(archive) as z:
            members = {n: z.read(n) for n in z.namelist() if not n.endswith('/')}
        prefix = ''
    else:
        with tarfile.open(archive) as z:
            assert all(m.isfile() or m.isdir() for m in z.getmembers()), 'No archive links'
            members = {m.name: z.extractfile(m).read() for m in z.getmembers() if m.isfile()}
        prefix = f'jades_eval-{version}/'
    for name, data in members.items():
        path = Path(name)
        assert not path.is_absolute() and '..' not in path.parts
        assert not set(path.parts) & {'.env', '.git', 'validation', 'reviews', 'tests', 'tools', '__pycache__', 'JADES_light', 'Truthful_eval'}
        assert b'/Users/' not in data, 'Local workspace path in distribution'
    for source in (root / 'src/jades').rglob('*'):
        if not source.is_file() or '__pycache__' in source.parts:
            continue
        name = prefix + str(source.relative_to(root if prefix else root / 'src')).replace(os.sep, '/')
        assert members.get(name) == source.read_bytes(), f'Source mismatch: {name}'
metadata = root / 'release-metadata'
metadata.mkdir(exist_ok=True)
(metadata / 'SHA256SUMS').write_text(''.join(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n' for p in files))
changelog = (root / 'CHANGELOG.md').read_text(encoding='utf-8')
section = changelog.split(f'## {version}\n', 1)[1].split('\n## ', 1)[0].strip()
(metadata / 'notes.md').write_text(f'## {version}\n\n{section}\n', encoding='utf-8')
print(json.dumps({'version': version, 'files': [p.name for p in files], 'checks': 'passed'}))

if os.environ.get("GITHUB_OUTPUT"):
    with open(os.environ["GITHUB_OUTPUT"], "a") as output:
        output.write(f"version={version}\n")
