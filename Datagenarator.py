from __future__ import print_function
import numpy as np

months=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
days=['Sun','Mon','Tue','Wed','Thu','Fri','Sat']
daytime=['AM','PM']
monthlen=[31,28,31,30,31,30,31,31,30,31,30,31]

def generate_time():
    m=np.random.randint(0,12)
    d=np.random.randint(0,7)
    t=np.random.randint(0,24)
    m_=months[m]
    ml=str(monthlen[m])
    m='m'+str(m)
    d_=str(days[d])
    d='w'+str(d)
    if(t>12):
        t_=str(daytime[1])
    else:
        t_=str(daytime[0])
    t='t'+str(t)
    input_tuple=[m,d,t]
    target_tuple=[m_,d_,t_,ml]
    order=np.random.permutation([0,1,2])
    in_string=""
    out_string=""
    for o in order:
        in_string+=str(input_tuple[o])+" "
        out_string+=str(target_tuple[o])+" "
    out_string+=ml
    print("Input %s Output %s"%(in_string,out_string))
    return in_string.rstrip(),out_string.rstrip()

def generate_time_data(count,corpus):
    f=open(corpus,"w")
    for c in range(count):
        i,o=generate_time()
        f.write(i+"\t"+o+"\n")
    f.close()

generate_time_data(5000,"time_corpus")
#generate_time()
