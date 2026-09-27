"""Verify the comparison checkpoint, reports, source and robot assets. No motion."""
import hashlib
import json
from pathlib import Path

here = Path(__file__).resolve().parent
root = here.parents[2]
manifest = json.loads((here / 'manifest.json').read_text())
checked = 0
for directory, entries in ((here, manifest['files']),
                           (root, manifest['runtime_source_sha256']),
                           (root, manifest['asset_sha256'])):
    for name, expected in entries.items():
        path = directory / name
        if not path.is_file():
            raise SystemExit(f'Missing required file: {path}')
        content = path.read_bytes()
        if directory == root and name in manifest['normalize_source_line_endings']:
            content = content.replace(b'\r\n', b'\n')
        actual = hashlib.sha256(content).hexdigest()
        if actual != expected:
            raise SystemExit(f'Hash mismatch: {path}\nExpected {expected}\nActual   {actual}')
        checked += 1
assert manifest['policy_sha256'] == manifest['files']['policy.pt']
print(f'PASS: {checked} package/source/asset hashes match.')
print(f"Policy: {manifest['policy_sha256']}")
print('Simulation comparison package; physical and actual-camera validation remain pending.')
