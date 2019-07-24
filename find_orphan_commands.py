import re
import sys
import glob
import string

verbose = True if '-v' in sys.argv else False
tex = glob.glob('*/*.tex')
acronyms = set()
uppercase = set(string.ascii_uppercase)

remove = ('\item',)
for fname in tex:
    f = open(fname)
    for num, line in enumerate(f):
        if '\\' in line:
            out = re.findall(r'\\[A-z]*', line)
            out = [x for x in out  if x + ' ' in line and x not in remove]
            if out:
                print(fname, num + 1, out)

