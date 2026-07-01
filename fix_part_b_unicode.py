"""Script to insert UTF-8 stdout fix into part_b_skeleton/run.py"""
import os

path = os.path.join('part_b_skeleton', 'run.py')
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

fix_line = (
    '\r\n# ITEM 6: prevent cp1252 UnicodeEncodeError on Windows\r\n'
    'try:\r\n'
    '    sys.stdout.reconfigure(encoding="utf-8", errors="replace")\r\n'
    'except AttributeError:\r\n'
    '    pass\r\n'
)

marker = 'if _REPO_ROOT not in sys.path:\r\n    sys.path.insert(0, _REPO_ROOT)\r\n'

if marker in content and fix_line not in content:
    content = content.replace(marker, marker + fix_line, 1)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print('Fix applied to part_b_skeleton/run.py')
elif fix_line in content:
    print('Already fixed.')
else:
    # Try LF version
    marker_lf = 'if _REPO_ROOT not in sys.path:\n    sys.path.insert(0, _REPO_ROOT)\n'
    fix_lf = (
        '\n# ITEM 6: prevent cp1252 UnicodeEncodeError on Windows\n'
        'try:\n'
        '    sys.stdout.reconfigure(encoding="utf-8", errors="replace")\n'
        'except AttributeError:\n'
        '    pass\n'
    )
    if marker_lf in content:
        content = content.replace(marker_lf, marker_lf + fix_lf, 1)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        print('Fix applied (LF variant) to part_b_skeleton/run.py')
    else:
        print('Pattern not found - manual fix needed')
        print(repr(content[1800:2100]))
