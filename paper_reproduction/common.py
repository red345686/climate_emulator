from pathlib import Path
from climate_emulator.utils import read_json,write_json,file_hash
PROJECT=Path(__file__).resolve().parents[1]
def verify(root, records):
    root = Path(root).resolve()
    for name, digest in records.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file() or file_hash(path) != digest:
            raise ValueError(f'Missing, changed or escaping artifact: {name}')


def records(root, paths):
    root = Path(root).resolve()
    return {Path(p).resolve().relative_to(root).as_posix(): file_hash(p) for p in paths}


def new_or_same(path, value):
    if path.exists() and read_json(path) != value:
        raise ValueError(f'Frozen record changed: {path}. Use a fresh output directory.')
    if not path.exists():
        write_json(path, value)


def freeze(output, identity, paths):
    new_or_same(output / 'protocol_lock.json', dict(fingerprint=identity,
                                                   artifacts=records(output, paths)))


def verify_frozen(output, identity):
    path = output / 'protocol_lock.json'
    if not path.is_file():
        raise ValueError('Run selection successfully for the ENTIRE matrix before final/evaluate')
    frozen = read_json(path)
    if frozen['fingerprint'] != identity:
        raise ValueError('Frozen protocol identity differs')
    verify(output, frozen['artifacts'])

