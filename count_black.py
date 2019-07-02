import subprocess
out = subprocess.check_output('gs -o - -sDEVICE=inkcov main.pdf', shell=True)
out = out.splitlines()

# Skip header:
ix = out.index([l for l in out if l.startswith(b'Processing')][0])

colour = []
bw_ct = 0


for l in out[ix + 1:]:
    l = l.decode()
    if 'Page' in l:
        page = int(l.split()[1])
    else:
        if all(float(x)==0 for x in l.split()[:3]):
            #print(page,'B&W')
            bw_ct += 1
        else:
            #print(page, 'Colour')
            colour.append(page)

print()
print('{} BW'.format(bw_ct))
print('{} Colour'.format(len(colour)))
print('{:.1f}% B&W'.format(100 * bw_ct / (bw_ct + len(colour))))

