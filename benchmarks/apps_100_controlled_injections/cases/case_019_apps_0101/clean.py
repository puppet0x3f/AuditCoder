def egcd(a, b):
    x,y, u,v = 0,1, 1,0
    while a != 0:
        q, r = b//a, b%a
        m, n = x-u*q, y-v*q
        b,a, x,y, u,v = a,r, u,v, m,n
    gcd = b
    return gcd, x, y


import math
n=int(input())
a=int(input())
b=int(input())
gcd,x,y=(egcd(a,b))


status=0
if((n%gcd)!=0):
    print("NO")
    #print("point1")

else:
    multiply=n/gcd
    x1=int(multiply*x)
    y1=int(multiply*y)
    #print("gcd and soln to n")
    #print(gcd,x1,y1)
    d1=b/gcd
    d2=a/gcd
    rangemin= int(math.ceil(-x1/d1))
    rangemax= int(y1//d2)
    #print("rangemin and rangemax")
    #print(rangemin,rangemax)
    if(rangemin>rangemax):
        print("NO")
        #print("point2")
    else:
        #print("YES")
        #solx=x1+rangemin*d1
        #soly=y1-rangemin*d2
        m=rangemin
        while(m<=rangemax):
            solx=x1+m*d1
            soly=y1-m*d2
            if(solx>=0 and soly>=0):
                print("YES")
                status=1
                print(str(int(solx))+" "+str(int(soly)))
                break
            m=m+1

        if(status==0):
            print("NO")
            #print("point3")
                
        
