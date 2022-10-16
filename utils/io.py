import json

def read_json(fpath):
    with open(fpath,'r') as f:
        res = json.load(f)
    return res

def save_json(data,fpath):
    with open(fpath,'w') as f:
        f.write(json.dumps(data))
