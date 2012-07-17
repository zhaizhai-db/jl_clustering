import re

fin = open('log.txt', 'r')
fout = open('fixedlog.txt', 'w')

p = ' ' + fin.read() + ' '
p = re.sub(r'[ :,\n]', ' ', p)
p = re.sub(' [a-zA-Z_]+', ' ', p)
p = re.sub(r'[ ]+', ' ', p)

fout.write(p)
