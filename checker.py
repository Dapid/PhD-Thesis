from pathlib import Path

files = Path('.').glob('**/*tex')

for f in files:
    for line in open(f):
        if line.count('"') != line.count('``'):
            print(f, line)
