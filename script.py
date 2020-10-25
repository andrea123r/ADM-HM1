#Say "Hello, World!" With Python

if __name__ == '__main__':
    my_string = "Hello, World!"
    print(my_string)

#Python If-Else

#!/bin/python

import math
import os
import random
import re
import sys



if __name__ == '__main__':
    n = int(raw_input().strip())
if n % 2 ==1:
    print("Weird")
elif n % 2 == 0 and 2 <= n <= 5:
    print("Not Weird")
elif n % 2 == 0 and 6 <= n <= 20:
    print("Weird")
else:
    print("Not Weird")

#Loops

if __name__ == '__main__':
    a=5
    b=9
    n=int(input())
    for i in range(n):
        print(i**2)

#Write a function

def is_leap(year):
    leap = False
    
    # Write your logic here
    
    year= 1990
    return year % 4 ==0 and (year % 400==0 or year %100!=0)
    return leap 
    year= 2000
def is_leap(year):
    return year % 4 ==0 and (year % 400==0 or year %100!=0)
    return leap

#Print Function

n=int(input())
for i in range(1, n+1):
   print(i,end="")

#Python: Division

if __name__ == '__main__':
    a = int(input())
    b = int(input())
    print(a//b)
    print(a/b)

#Arithmetic Operators

if __name__ == '__main__':
    a = int(input())
    b = int(input())
a+b
print(a+b)
a-b
print(a-b)
a*b
print(a*b)

#Find the Runner-Up Score!

if __name__ == '__main__':
    n = int(input())
    arr = list(map(int, input().split()))
    
    g=max (arr)
    while max (arr)==g:
        arr.remove(max (arr))

    print (max(arr))

#Finding the percentage

if __name__ == '__main__':
    n = int(input())
    student_marks = {}
    for _ in range(n):
        name, *line = input().split()
        scores = list(map(float, line))
        student_marks[name] = scores
    query_name = input()

    s=0
    for i in student_marks[query_name]:
        s=s+i
    print("{0:.2f}".format(s/3))

#Lists

if __name__ == '__main__':
    N = int(input())
    arr =[]
    for i in range(N):
        s=input().split()
        for i in range(1,len(s)):
            s[i]= int(s[i])

        if s[0]== "append":
            arr.append(s[1])
            
        elif s[0] == "insert":
            arr.insert(s[1],s[2])  
        elif s[0] == "remove":
            arr.remove(s[1])  
        elif s[0] == "pop":
            arr.pop()
        elif s[0] == "sort":
            arr.sort()
        elif s[0] == "reverse":
            arr.reverse()
        elif s[0] == "print":
            print(arr)           
           

#Tuples

if __name__ == '__main__':
    n=int(input())
    integer_list= tuple(map(int, input().split()))
    t=integer_list
    print(hash(t))  

#List Comprehensions

if __name__ == '__main__':
    x = int(input())
    y = int(input())
    z = int(input())
    n = int(input())
    print ([[a,b,c] for a in range(0, x+1) for b in range(0, y+1) for c in range(0, z+1) if a+b+c !=n])

#Nested Lists

marksheet=[]
scoresheet=[]
if __name__ == '__main__':
    for _ in range(int(input())):
        name = input()
        score = float(input())
        marksheet+=[[name, score]]
        scoresheet+=[score]
    y=sorted(set(scoresheet))[1]
    
    for name, score in sorted(marksheet):
        if score==y:
            print(name)

#String Split and Join

def split_and_join(line):
    # write your code here
    return '-'.join(line.split(" "))

#What's your name?

def print_full_name(a, b):
    print ("Hello %s %s! You just delved into python."% (a,b))

#Mutations

def mutate_string(string, position, character):
    return string[:position]+character+string[position+1:]

#Find a string

def count_substring(string, sub_string):
    l=len(string)
    l1=len(sub_string)
    c=0
    for i in range(l-l1+ 1):
        if(string[i:(i+l1)]== sub_string):
            c=c+1
    return c

#String Validators

if __name__ == '__main__':
    s = raw_input()
    print(any([i.isalnum() for i in s]))
    print(any([i.isalpha() for i in s]))
    print(any([i.isdigit() for i in s]))
    print(any([i.islower() for i in s]))
    print(any([i.isupper() for i in s]))

#Text Alignment

thickness = int(raw_input()) #This must be an odd number
c = 'H'

#Top Cone
for i in range(thickness):
    print (c*i).rjust(thickness-1)+c+(c*i).ljust(thickness-1)

#Top Pillars
for i in range(thickness+1):
    print (c*thickness).center(thickness*2)+(c*thickness).center(thickness*6)

#Middle Belt
for i in range((thickness+1)/2):
    print (c*thickness*5).center(thickness*6)    

#Bottom Pillars
for i in range(thickness+1):
    print (c*thickness).center(thickness*2)+(c*thickness).center(thickness*6)    

#Bottom Cone
for i in range(thickness):
    print ((c*(thickness-i-1)).rjust(thickness)+c+(c*(thickness-i-1)).ljust(thickness)).rjust(thickness*6)

#Text Wrap

def wrap(string, max_width):
    return(textwrap.fill(string,max_width))

#Designer Door Mat

R, C =map(int,input().split(' '))

for i in range(1, R,2):
    print((".|."*i).center(C, '-'))

print("WELCOME".center(C, '-'))

for i in range(R-2, -1, -2):
    print((".|."*i).center(C, '-'))

#String Formatting

def print_formatted(number):
    w= len(str(bin(number))[2:])
    for i in range(1,number+1):
        print(str(i).rjust(w, (' ')), oct(i)[2:].rjust(w, (' ')), hex(i)[2:].upper().rjust(w, (' ')), bin(i)[2:].rjust(w, (' ')))

#Alphabet Rangoli

def print_rangoli(size):
    n=size
    l1=list(map(chr,range(9, 17)))
    x=l1[n-1::n]+l1[1:n]
    m=len('-'.join(x))
    for i in range(n):
        s='-'.join(chr(ord('a')+n-j-1) for j in range(i+1))
        print((s+s[::-1][1:]).center(n*4-3,'-'))
    for i in range(n-1):
        s='-'.join(chr(ord('a')+n-j-1) for j in range(n-i-1))
        print((s+s[::-1][1:]).center(n*4-3,'-'))
        
#The Minion Game

def minion_game(string):
    s1=0
    s2=0
    vow='AEIOU'
    for i in range(len(s)):
        if s[i] not in vow:
            s1=s1+(len(s)-i)
        else:
            s2=s2+(len(s)-i)
    if s1>s2:
        print("Stuart",s1)
    elif s2>s1:
        print("Kevin",s2)
    else:
        print("Draw")

#Capitalize

import string
def solve(s):
   return string.capwords(s, ' ')
   
#Merge the Tools!

def merge_the_tools(string, k):
    c=0
    s=''
    for i in string:
        if i not in s:
            s=s+i
        c+=1
        if (c==k):
            print(s)
            c=0
            s=''

#sWAP cASE

def change(s):
    if str.islower(s):
        return str.upper(s)
    else:
        return str.lower(s)

def swap_case(s):
    return ''.join(map(change,s))

#Introduction to Sets

def average(array):
    array=set(array)

    return sum(array)/len(array)
    
# Symmetric difference

n=int(input())
s1=set(map(int, input().split()))
m=int(input())
s2=set(map(int, input().split()))

s3=s1.union(s2)

l= []
for i in s3:
    if i in s1 and i in s2:
        pass
    else:
        l.append(i)

#Set.add()

n=int(input())
x=[input() for i in range(n)]
l=set(x)
print(len(l))

#Set .discard(), .remove() & .pop()

input()
s = set(map(int, input().split()))
n=int(input())
for i in range(n):
    commands=input().split()
    if len(commands)>1:
        a=int(commands[1])
    if commands[0]=="pop":
        s.pop()
    if commands[0]=="remove":
        s.remove(a)
    if commands[0]=="discard":
        s.discard(a)
print(sum(s))

#Set .union() Operation

n=int(input())
s1=set(map(int, input().split()))
b=int(input())
s2=set(map(int, input().split()))
print(len(s1.union(s2)))

#Set .intersection() Operation

n=int(input())
s1=set(map(int, input().split()))
b=int(input())
s2=set(map(int, input().split()))
print(len(s1.intersection(s2)))

#Set .difference() Operation


e=int(input())
s1=set(map(int, input().split()))
f=int(input())
s2=set(map(int, input().split()))
print(len(s1.difference(s2)))

#Set .symmetric_difference() Operation

e=int(input())
s1=set(map(int, input().split()))
f=int(input())
s2=set(map(int, input().split()))
print(len(s1.symmetric_difference(s2)))

#Set Mutations

(input())
s1=set(map(int, input().split()))
n=int(input())


for i in range(n):
    global output
    r=input().split()
    operation=r[0]
    s2=set(map(int, input().split()))
    if operation == "intersection_update":
        s1.intersection_update(s2)
    elif operation == "update":
        s1.update(s2)
    elif operation == "symmetric_difference_update":
        s1.symmetric_difference_update(s2)
    elif operation == "difference_update":
        s1.difference_update(s2)
    output=sum(s1)
print(output)

#The Captain's Room

k=int(input())
rooms=list(map(int, input().split()))
a=set()
c=set()
for room in rooms:
    if room not in a:
        a.add(room)
        c.add(room)
    else:
        c.discard(room)
c=list(c)
print(c[0])

#Check subsets

t=int(input())
def check():
   a=int(input())
   s1=set(map(int, input().split()))
   b=int(input())
   s2=set(map(int, input().split()))

   l=[]
   for i in s1:
       if i in s2:
           l.append(i)
   l=set(l)
   if l==s1:
        print(True)
   else:
        print(False)

for i in range(t):
    check()

#Check Strict Superset

a=set(map(int, input().split()))
n=int(input())
l=[]
def check():
    sub=set(map(int, input().split()))
    if sub.issubset(a):
       if len(a)==len(sub):
          l.append(0)
       else:
          l.append(1)
    else:
       l.append(0)


for i in range(n):
    check()
if all(l)==1:
    print(True)
else:
    print(False)

#collections.Counter()

from collections import Counter
def customers():
    l.append(list(map(int, input().split())))
l=[]
   
x=int(input())
shoesize=list(map(int, input().split()))
c=int(input())

for i in range(c):
    customers()
earnings=0
dict=Counter(shoesize)
for i, p in l:
    if i in dict.keys() and dict[i]>0:
        earnings+=p
        dict[i]=dict[i]-1
print(earnings)

#Collections.namedtuple()

from collections import namedtuple
n=int(input())
columns=input().split()
students=namedtuple('students', columns)
s=0
for i in range(n):
    column1, column2, column3, column4=input().split()
    student=students(column1,column2,column3,column4)
    s+=int(student.MARKS)
print('{:.2f}'.format(s/n))

#Collections.OrderedDict()

from collections import OrderedDict
n=int(input())
dict=OrderedDict()
for i in range(n):
    item, space, price=input().rpartition(" ")
    dict[item]=dict.get(item,0)+int(price)
for item, price in dict.items():
    print(item, price)

#Collections.deque()

from collections import deque
d=deque()
n=int(input())
for i in range(n):
    l=input().split()
    command=l[0]
    if len(l)>1:
        s=l[1]
    if command=="append":
        d.append(s)
    elif command=="pop":
        d.pop()
    elif command=="popleft":
        d.popleft()
    elif command=="appendleft":
        d.appendleft(s)

#Company Logo

#!/bin/python3

import math
import os
import random
import re
import sys
from collections import Counter

s =sorted(input())
c=Counter(s).most_common(3)
for i, j in c:
   print(i, j)

#Piling Up!

from collections import deque
num_t=int(input())

for _ in range(num_t):
    _, queue=input(), deque(map(int, input().split()))

    for curr_cube_len in reversed(sorted(queue)):
        if queue[-1]==curr_cube_len:queue.pop()
        elif queue[0]==curr_cube_len:queue.popleft()
        else:
           print("No")
           break
    else: print("Yes")

#Word Order

from collections import Counter
n=int(input())
i=list()
for _ in range(n):
    i.append(input())
num_w=Counter(i)

print(len(num_w))
print(*num_w.values())

#Calendar Module

import calendar
m, d, y=map(int, input().split())
days={0:'MONDAY', 1:'TUESDAY', 2:'WEDNESDAY', 3:'THURSDAY', 4:'FRIDAY', 5:'SATURDAY', 6:'SUNDAY'}
print(days[calendar.weekday(y, m, d)])

#Exceptions

t=int(input())
for i in range(t):
    try:
        a,b=map(int, input().split())
        print(a//b)
    except ZeroDivisionError as e:
        print("Error Code:",e)
    except ValueError as e:
        print("Error Code:", e)

#ginortS

l=[]
s=sorted(input())
lowercase=""
upper=""
digits=""
odd=""
even=""
for i in s:
    if i.islower():
        lowercase+=i
    elif i.isupper():
        upper+=i
    elif int(i) % 2!=0:
        odd+=i
    elif int(i) % 2==0:
        even+=i

print(lowercase+upper+digits+odd+even)

#Map and Lambda Function

def fibonacci(n):
    a=0
    b=1
    l=[]
    if n==0:
        pass
    elif n==1:
        l.append(a)
    else:
        l.append(a)
        l.append(b)
        for i in range(2, n):
            c=a+b
            a=b
            b=c
            l.append(c)
    return l
cube=lambda x: x**3  

#Detect Floating Point Number

import re
n=int(input())
pattern= r'^[+-]?[0-9]*\.[0-9]+$'
for i in range(n):
    s=input()
    print(bool(re.match(pattern, s)))

#Re.split()

regex_pattern = r"[.,]"	# Do not delete 'r'.

#Group(), Groups() & Groupdict()

import re
s=input()
pattern=r'([a-z A-Z 0-9])\1'
n=re.search(pattern, s)
if n:
    print(n.groups()[0])
else:
    print(-1)

#Re.findall() & Re.finditer()

import re
def find_substrings(s):
    vowel='[aeiou]'
    consonant='[qwrtypsdfghjklzxcvbnm]'

    return re.findall(r'{consonant}({vowel}{{2,}})(?={consonant})'.format(vowel=vowel, consonant=consonant),s,re.IGNORECASE)

def main():
    s=input()
    substrings=find_substrings(s)
    if substrings:
        print(*substrings, sep='\n')
    else:
        print(-1)

if __name__ == '__main__':
    main()

#Re.start() & Re.end()

import re
s=input()
k=input()
pattern=re.compile(k)
match=pattern.search(s)
if not match: print("(-1, -1)")
while match:
    print('({}, {})'.format(match.start(), match.end()-1))
    match=pattern.search(s, match.start()+1)
    
#Regex Substitution

import re
n=int(input())

for i in range(n):
    print(re.sub(r'(?<= )\|\|(?= )', "or", re.sub(r'(?<= )&&(?= )', "and",input())))

#Validating Phone Numbers

import re
n=int(input())
pattern="^[987]"
for i in range(n):
    number=input()
    if len(number)==10 and number.isdigit():
        val=re.findall(pattern, number)
        if len(val)==1:
            print("YES")
        else:
            print("NO")
    else:
        print("NO")
        
#Hex Color Code

import re
n=int(input())

for _ in range(n):
    matches=re.findall(r':?.(#[0-9A-Fa-f]{6}|#[0-9a-fA-F]{3})', input())
    if matches:
        print(*matches, sep="\n")

#HTML Parser - Part 1

from html.parser import HTMLParser
import re
def read_int():return int(input())
def read_int_words(): return map(int, input().split())
n=int(input())


class myHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print('Start :', tag)
        for attr in attrs:
            print('->',' > ' .join(map(str, attr)))
    def handle_endtag(self, tag):
        print('End   :', tag)
    def handle_startendtag(self, tag, attrs):
        print('Empty :', tag)
        for attr in attrs:
           print('->',' > ' .join(map(str, attr))) 

parser=myHTMLParser()
for i in range(n): parser.feed(input())

#Detect HTML Tags, Attributes and Attribute Values

from html.parser import HTMLParser

class myHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print(tag)
        for at in attrs:
            print("-> {} > {}".format(at[0], at[1]))
    def handle_startendtag(self, tag, attrs):
        print(tag)
        for at in attrs:
            print("-> {} > {}".format(at[0], at[1]))

n=int(input())
html=""
for i in range(n):
    html+=input().rstrip()
    html+='\n'

parser=myHTMLParser()
parser.feed(html)    
parser.close()

#Validating UID

import re
t=int(input())
for i in range(t):
    UID=input().strip()
    if UID.isalnum() and len(UID)==10:
       if bool(re.search(r'(.*[A-Z]){2,}', UID)) and bool(re.search(r'(.*[0-9]){3,}',UID)):
          if re.search(r'.*(.).*\1+.*', UID):
             print('Invalid')
          else:
             print('Valid')
       else:
           print('Invalid')
    else:
        print('Invalid')

#Validating Postal Codes

regex_integer_in_range = r"[1-9][\d]{5}$"	# Do not delete 'r'.
regex_alternating_repetitive_digit_pair = r"(?=(\d)\d\1)"	# Do not delete 'r'.

#Matrix Script

import math
import os
import random
import re
import sys


first_multiple_input = input().rstrip().split()

n = int(first_multiple_input[0])

m = int(first_multiple_input[1])

matrix = []

for _ in range(n):
    matrix_item = input()
    matrix.append(matrix_item)

string=""
for y in range(m):
    for x in range(n):
        string+=matrix[x][y]
pattern=re.compile(r"(\w)(\W+)(\w)")
sub=pattern.sub(r"\1 \3", string)
print(sub)

#XML1 - Find the Score

def get_attr_number(node):
    return(len(node.attrib)+sum(get_attr_number(s) for s in node))

#XML2 - Find the Maximum Depth

maxdepth = 0
def depth(elem, level):
    global maxdepth
    level+=1
    for s in elem:
        depth(s, level)
    if level>=maxdepth:
        maxdepth=level

#Standardize Mobile Number Using Decorators

def wrapper(f):
    def fun(l):
        res_n=[]
        
        for number in l:
            m=""
            m=number[::-1][0:10][::-1]
            m=" ".join(["+91"]+[m[0:5], m[5:]])
            res_n.append(m)
        f(res_n)

    return fun

#Decorators 2 - Name Directory

def person_lister(f):
    def inner(people):
        people=sorted(people, key=lambda x:int(x[2]))
        ret=[]
        for p in people:
            ret.append(f(p))
        return ret
    return inner

#Arrays

def arrays(arr):
    return(numpy.array(arr[::-1], float))

#Shape and Reshape

import numpy
print(numpy.array(input().split(), int).reshape(3, 3))

#Transpose and Flatten

import numpy
n, m=map(int, input().split())
array=numpy.array([input().strip().split() for _ in range(n)], int)
print(array.transpose())
print(array.flatten())

#Zeros and Ones

import numpy
num=tuple(map(int, input().split()))
print(numpy.zeros(num, dtype=numpy.int))
print(numpy.ones(num, dtype=numpy.int))

#Eye and Identity

import numpy
numpy.set_printoptions(sign=' ')
print(numpy.eye(*map(int, input().split())))

#Array Mathematics

import numpy
n, m=map(int, input().split())
a=(numpy.array([input().split() for _ in range(n)], dtype=int))
b=(numpy.array([input().split() for _ in range(n)], dtype=int))
print(numpy.add(a, b))
print(numpy.subtract(a, b))
print(numpy.multiply(a, b))
print(a//b)
print(numpy.mod(a, b))
print(numpy.power(a, b))

#Floor, Ceil and Rint

import numpy
numpy.set_printoptions(sign=' ')
a=numpy.array(input().split(), float)
print(str(numpy.floor(a)))
print(str(numpy.ceil(a)))
print(str(numpy.rint(a)))

#Sum and Prod

import numpy
n, m=map(int, input().split())
array=numpy.array([input().split() for _ in range(n)], int)
s=(numpy.sum(array, axis=0))
print(numpy.prod(s, axis=0))

#Min and Max

import numpy
n, m= map(int, input().split())
array=numpy.array([input().split() for _ in range(n)], int)
s=numpy.min(array, axis=1)
print(numpy.max(s, axis=0))

#Mean, Var and Std

import numpy
n, m=map(int, input().split())
array=numpy.array([input().split() for _ in range(n)], int)
numpy.set_printoptions(legacy='1.13')
print(numpy.mean(array, axis=1))
print(numpy.var(array, axis=0))
print(numpy.std(array))

#Inner and Outer

import numpy
array=numpy.array(input().split(), int)
b=numpy.array(input().split(), int)
print(numpy.inner(array, b))
print(numpy.outer(array, b))

#Polynomials

import numpy
p=list(map(float, input().split()))
r=input()
print(numpy.polyval(p, int(r)))

#Linear Algebra

import numpy
n=int(input())
array=[list(map(float, input().split())) for _ in range(n)]
print(round(numpy.linalg.det(array), 2))

#Number Line Jumps

#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the kangaroo function below.
def kangaroo(x1, v1, x2, v2):
    if x1<x2 and v1<v2:
        return('NO')
    else:
        if v1!=v2 and (x2-x1)%(v1-v2)==0:
            return('YES')
        else:
            return('NO')


if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    x1V1X2V2 = input().split()

    x1 = int(x1V1X2V2[0])

    v1 = int(x1V1X2V2[1])

    x2 = int(x1V1X2V2[2])

    v2 = int(x1V1X2V2[3])

    result = kangaroo(x1, v1, x2, v2)

    fptr.write(result + '\n')

    fptr.close()

#Viral Advertising

import math
import os
import random
import re
import sys

# Complete the viralAdvertising function below.
def viralAdvertising(n):
    total_likes=0
    shared=5

    for i in range(n):
        like=shared//2
        total_likes+=like
        shared=like*3

    return(total_likes)

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input())

    result = viralAdvertising(n)

    fptr.write(str(result) + '\n')

    fptr.close()


#Recursive Digit Sum

#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the superDigit function below.
def superDigit(n, k):
    def sum_digit(v):
        if v<10:
            return v
        s=sum(int(i) for i in str(v))
        return sum_digit(s)
    x=sum_digit(int(n))
    return sum_digit(x*k)

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    nk = input().split()

    n = nk[0]

    k = int(nk[1])

    result = superDigit(n, k)

    fptr.write(str(result) + '\n')

    fptr.close()

#Insertion Sort - Part 1

#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the insertionSort1 function below.
def insertionSort1(n, arr):
    e=arr[-1]
    for i in range(n-2, -1, -1):
        if arr[i]>e:
            arr[i+1]=arr[i]
            print(" ".join(str(j) for j in arr))
        else:
            arr[i+1]=e
            print(" ".join(str(j) for j in arr))
            return
    arr[0]=e
    print(" ".join(str(j) for j in arr))
    return

if __name__ == '__main__':
    n = int(input())

    arr = list(map(int, input().rstrip().split()))

    insertionSort1(n, arr)

#Incorrect Regex

import re
t=int(input())
def isvalidregex(regex):
    try: re.compile(regex)
    except re.error: return False
    return True
for i in range(t):
    print(isvalidregex(input()))

#Input()

x,k=map(int, input().split())
P=input()
f=lambda x:eval(P)
print(f(x)==k)

#Python Evaluation

(eval(input()))

#No Idea!

n_m=map(int, input().split())
n= map(int, input().split())
a=set(map(int, input().split()))
b=set(map(int, input().split()))
happiness=0
for i in n:
    if i in a:
        happiness+=1
    elif i in b:
        happiness-=1

print(happiness)

#DefaultDict Tutorial

from collections import defaultdict
dict=defaultdict(list)
n, m=(map(int, input().split()))
l=[]
for i in range(0, n):
    dict[input()].append(i+1)
for i in range(0, m):
    l.append(input())
for i in l:
    if i in dict:
        print(" ".join(map(str,dict[i])))
    else:
        print('-1')

#Time Delta

#!/bin/python3

import math
import os
import random
import re
import sys
from datetime import datetime
# Complete the time_delta function below.
def time_delta(t1, t2):
   a=datetime.strptime(t1, "%a %d %b %Y %H:%M:%S %z")

   b=datetime.strptime(t2, "%a %d %b %Y %H:%M:%S %z")
   return(str(int(abs(a-b).total_seconds())))
        
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    t = int(input())

    for t_itr in range(t):
        t1 = input()

        t2 = input()

        delta = time_delta(t1, t2)

        fptr.write(delta + '\n')

    fptr.close()


#Zipped!

nx=list(map(int, input().split()))
n, x=nx[0], nx[1]
marks=[]
for i in range(x):
    marks.append(list(map(float, input().split())))
s=list(zip(*marks))
for i in s:
    print((sum(i)/len(i)))

#Validating Roman Numerals

thousands=r'M{0,3}'
hundreds=r'(?:D?C{0,3}|CM|CD)'
tens=r'(?:L?X{0,3}|XC|XL)'
digits=r'(?:V?I{0,3}|IX|IV)'
regex_pattern = r"^"+thousands+hundreds+tens+digits+"$"	

#Validating and Parsing Email Addresses
import re
import email.utils
n=int(input())
for i in range(n):
    line=input()
    name,email=line.split(" ")
    pattern="<[a-z][a-zA-Z0-9\-\.\_]+@[a-zA-Z]+\.[a-zA-Z]{0,3}>"
    if(bool(re.match(pattern, email))):
        print(name,email)

#HTML Parser - Part 2
from html.parser import HTMLParser

class MyHTMLParser(HTMLParser):
    def handle_comment(self, data):
        if data!='\n':
           if '\n' not in data:
              print(">>> Single-line Comment")
              print(data)
           else:
              print(">>> Multi-line Comment")
              print(data)

    def handle_data(self, data):
        if data!='\n':
            print(">>> Data")
            print(data)

  
html = ""       
for i in range(int(input())):
    html += input().rstrip()
    html += '\n'
    
parser = MyHTMLParser()
parser.feed(html)
parser.close()



#Validating Credit Card numbers
import re
TESTER=re.compile(r"^"
r"(?!.*(\d)(-?\1){3})"
r"[456]"
r"\d{3}"
r"(?:-?\d{4}){3}"
r"$")
for _ in range(int(input().strip())):
    print("Valid" if TESTER.search(input().strip()) else "Invalid")

#Concatenate
import numpy
n, m, p =[int(i) for i in input().split()]
array1=numpy.array([input().split() for i in range(n)], int)
array2=numpy.array([input().split() for j in range(m)], int)
print(numpy.concatenate((array1, array2), axis=0))

#Birthday Cake Candles
#!/bin/python

import math
import os
import random
import re
import sys

#
# Complete the 'birthdayCakeCandles' function below.
#
# The function is expected to return an INTEGER.
# The function accepts INTEGER_ARRAY candles as parameter.
#

def birthdayCakeCandles(candles):
    n=len(candles)
    maxnum=0
    count=0
    for i in range(n):
        if candles[i]>maxnum:
            maxnum=candles[i]
            count=1
        elif candles[i]==maxnum:
            count+=1
    return(count)
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    candles_count = int(raw_input().strip())

    candles = map(int, raw_input().rstrip().split())

    result = birthdayCakeCandles(candles)

    fptr.write(str(result) + '\n')

    fptr.close()

#Insertion Sort - Part 2
#!/bin/python

import math
import os
import random
import re
import sys

# Complete the insertionSort2 function below.
def insertionSort2(n, arr):
    for q in range(1, len(arr)):
        for i in range(q):
            if (arr[q]<arr[i]):
                s=arr[q]
                arr[q]=arr[i]
                arr[i]=s
        print(' '.join(str(j) for j in arr))

if __name__ == '__main__':
    n = int(raw_input())

    arr = [int(i) for i in raw_input().strip().split()]

    insertionSort2(n, arr)

#Dot and Cross
import numpy
n=int(input())
a=numpy.array([input().split() for _ in range(n)],int)
b=numpy.array([input().split() for _ in range(n)], int)
print(numpy.dot(a,b))
