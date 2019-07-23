import re
import sys
import glob
import string

verbose = True if '-v' in sys.argv else False
tex = glob.glob('*/*.tex')
acronyms = set()
uppercase = set(string.ascii_uppercase)

for fname in tex:
    f = open(fname)
    for num, line_ in enumerate(f):
        line = line_.split('%', 1)[0].strip()
        words = re.split(' |,|\.|-|:|;|\(|\)|\[|\]|{|}|~', line)
        for w in words:
            if '$' in w: continue
            if len(w) < 3: continue
            if len(w) > 30: continue
            if w.startswith('\\'):continue
            if 'ReLU' in w: continue
            if sum(1 for x in w if x in uppercase) > 2: 
                if verbose:
                    print(fname, 1 + num,  line_)
                acronyms.add(w)

print(sorted(list(acronyms)))

