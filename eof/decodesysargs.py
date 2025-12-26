args={"field":"delT","em":"ps"}  # default arguments

import sys
arg0=sys.argv[1].split(",")
for s in arg0:
    s1=s.split('=',1)
    args[s1[0]]=s1[1];

