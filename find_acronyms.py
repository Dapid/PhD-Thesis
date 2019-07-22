import re
import glob
import string

tex = glob.glob('*/*.tex')
acronyms = set()
uppercase = set(string.ascii_uppercase)

for fname in tex:
    f = open(fname)
    for line in f:
        line = line.split('%', 1)[0].strip()
        words = re.split(' |,|\.|-|:|;|\(|\)|\[|\]|{|}|\\\\|~', line)
        for w in words:
            if '$' in w: continue
            if len(w) < 3: continue
            if len(w) > 30: continue
            if w.startswith('\\'):continue
            if sum(1 for x in w if x in uppercase) > 2: 
                acronyms.add(w)

print(sorted(list(acronyms)))

