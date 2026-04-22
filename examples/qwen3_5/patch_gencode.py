"""Patch gencode files: 4K seq_len -> 128K seq_len.

Only modifies forward/segment methods (not __init__ weight dims).
Replaces seq_len-related shapes:
  4096 (full seq) -> 131072
  4095 (full seq - 1, MTP) -> 131071
  1024 (TP-split seq = 4096/4) -> 32768 (131072/4)
"""
import re
import glob

GENCODE_PATTERN = '.nnscaler/_parallel_modules/__main__/WrapperModel/_/gencode*.py'

# Replacements: (pattern, replacement) — applied only in forward section
REPLACEMENTS = [
    # arange for position ids (backbone and MTP)
    ('end=4096', 'end=131072'),
    ('end=4095', 'end=131071'),
    # flash attention query_length (positional and keyword arg forms)
    (', 4096, True,', ', 131072, True,'),
    ('query_length=4095', 'query_length=131071'),
    # reshape/view with full seq_len
    ('shape=(1, 4096, -1)', 'shape=(1, 131072, -1)'),
    ('shape=(1, 4096, -1, 128)', 'shape=(1, 131072, -1, 128)'),
    # reshape/view with MTP seq_len (4095 = seq-1)
    ('shape=(1, 4095, -1)', 'shape=(1, 131071, -1)'),
    ('size=(1, 4095, -1,', 'size=(1, 131071, -1,'),
    # reshape/view with TP-split seq_len (4096/4=1024 -> 131072/4=32768)
    ('shape=(1, 1024, -1,', 'shape=(1, 32768, -1,'),
    ('shape=(1, 1024, -1)', 'shape=(1, 32768, -1)'),
    ('size=(1, 1024, -1,', 'size=(1, 32768, -1,'),
    ('size=(1, 1024, -1)', 'size=(1, 32768, -1)'),
]

def patch_file(filepath):
    with open(filepath) as f:
        lines = f.readlines()

    # Find where __init__ ends (first def after __init__)
    init_end = 0
    in_init = False
    for i, line in enumerate(lines):
        if 'def __init__' in line:
            in_init = True
        elif in_init and line.strip().startswith('def '):
            init_end = i
            break

    if init_end == 0:
        print(f'  WARNING: could not find forward section in {filepath}')
        return 0

    total_changes = 0
    for i in range(init_end, len(lines)):
        original = lines[i]
        modified = original
        for old, new in REPLACEMENTS:
            if old in modified:
                modified = modified.replace(old, new)
        if modified != original:
            lines[i] = modified
            total_changes += 1

    with open(filepath, 'w') as f:
        f.writelines(lines)

    return total_changes

files = sorted(glob.glob(GENCODE_PATTERN))
print(f'Found {len(files)} gencode files')
for f in files:
    n = patch_file(f)
    print(f'  {f}: {n} lines changed')
