import os
from pathlib import Path

def replace_checkmarks():
    root_dir = Path("d:/BAH/mega-heracross")
    for py_file in root_dir.rglob("*.py"):
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if '[OK]' in content or '\u2713' in content:
                new_content = content.replace('[OK]', '[OK]').replace('\u2713', '[OK]')
                
                with open(py_file, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                print(f"Updated {py_file}")
        except Exception as e:
            print(f"Failed {py_file}: {e}")

if __name__ == "__main__":
    replace_checkmarks()
