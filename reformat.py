import sys

for line in sys.stdin:
    line = line.strip()
    abc,d = line.split(" : x  =>  x = ")
    ab,c = abc.split(" :: ")
    a,b = ab.split(" : ")
    print("\t".join([a,b,c,d]))
