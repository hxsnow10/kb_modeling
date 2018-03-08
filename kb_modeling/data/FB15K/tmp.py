# encoding=utf-8
import sys

def transform(input_path, output_path):
    ii=open(input_path)
    oo=open(output_path,'w')
    ii.readline()
    for line in ii:
        name,id_=line.strip().split('\t')
        oo.write(id_+'\t'+name+'\n')
    oo.close()

transform(sys.argv[1], sys.argv[2]) 
