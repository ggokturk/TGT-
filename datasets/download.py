#!/usr/bin/env python3
import sys
import urllib.request
import gzip
import zipfile
from pathlib import Path
import io

def download_gz(url):
    with urllib.request.urlopen(url) as f:
        if url.endswith(".gz"):
            data = gzip.decompress(f.read()).decode()

        elif url.endswith(".zip"):
            z = zipfile.ZipFile(io.BytesIO(f.read()))
            data = z.read(z.infolist()[0]).decode()
            data = data.replace(','," ")        
        return data


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Missing the url argument, please provide a file that contains urls in each line", file=sys.stderr)
        exit(-1)
    
    urlpath = sys.argv[1]
    urlpairs = map(lambda x:x.split(), open(urlpath).readlines())
    
    for name, url in urlpairs:
        print(name, url)
        path = "./"+name
        Path(path).mkdir(parents=True, exist_ok=True)
        graph_path = path+"/graph.txt"
        if Path(graph_path).is_file():
            print("File ", graph_path, " exists")
            continue
        data = download_gz(url)
        data = data.split("\n")
        firstline = True
        firstline_wrote = False
        index = {}
        nextid = 0
        with open(graph_path,"w") as out: 
            for line in data:
                if line.startswith('#'):
                    continue
                if firstline == True:
                    firstline = False
                    continue
                if firstline_wrote:
                    out.write("\n")
                else:
                    firstline_wrote = True
                row = line.split()
                if len(row)==0:
                    continue
                if len(row)!=2 :
                    print("err:", line)
                s,t = row[0],row[1]
                if s not in index:
                    index[s] = nextid
                    nextid += 1
                if t not in index:
                    index[t] = nextid
                    nextid += 1

                out.write(str(index[s]))
                out.write('\t')
                out.write(str(index[t]))