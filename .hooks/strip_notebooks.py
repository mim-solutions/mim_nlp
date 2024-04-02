#!/usr/bin/env python3
import argparse
import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Optional

LOG = logging.getLogger('strip_notebooks')


def get_paths_from_command(*popenargs, **kwargs) -> list[Path]:
    output = subprocess.check_output(*popenargs, **kwargs)
    return list(map(Path, output.decode().splitlines(keepends=False)))


def process(git: Path, ipynb_path: Path, stripped_path: Path, no_index: bool):

    if not ipynb_path.exists():
        if stripped_path.exists() and no_index is False:
            LOG.info('Removing %s', stripped_path)
            subprocess.check_call([git, 'rm', stripped_path])
        return

    LOG.info('Stripping %s to %s', ipynb_path, stripped_path)
    with ipynb_path.open(encoding='utf-8') as f_in:
        data = json.load(f_in)
    stripped_path.parent.mkdir(parents=True, exist_ok=True)
    with stripped_path.open('w') as f_out:
        for cell in data['cells']:
            cell_type = cell['cell_type']
            source = cell['source']
            if cell_type in ('markdown', 'raw'):
                print('"""', file=f_out)
                for line in source:
                    print(line.rstrip('\n'), file=f_out)
                print('"""', file=f_out)
            elif cell_type == 'code':
                for line in source:
                    print(line.rstrip('\n'), file=f_out)
                print('# ---', file=f_out)
            else:
                raise ValueError('Cell type', cell_type)
    if no_index is False:
        LOG.info('Adding %s', stripped_path)
        subprocess.check_call([git, 'add', stripped_path])


def main(
        all_: bool,
        no_index: bool,
        output: Optional[Path],
        paths: list[Path],
):
    logging.basicConfig(level=logging.INFO)

    git = Path(shutil.which('git'))
    repo_path = get_paths_from_command([git, 'rev-parse',  '--show-toplevel'])[0]
    output = output or (repo_path / 'stripped')

    if all_:
        paths = get_paths_from_command([git, 'ls-files'])
    elif not paths:
        paths = get_paths_from_command([git, 'diff', '--cached', '--name-only', '--no-renames'])

    for path in paths:
        if not path.suffix == '.ipynb':
            continue
        stripped_path = (
            (output / (path.absolute().relative_to(repo_path)))
            .with_stem(path.stem + '_stripped')
            .with_suffix('.py')
        )
        process(git=git, ipynb_path=path, stripped_path=stripped_path, no_index=no_index)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--all', action='store_true', dest='all_')
    parser.add_argument('--no-index', action='store_true', help='do not modify git index')
    parser.add_argument('--output', type=Path)
    parser.add_argument('paths', type=Path, nargs='*')
    main(**vars(parser.parse_args()))
