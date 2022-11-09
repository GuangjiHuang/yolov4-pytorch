import yaml
import re
import os

def get_config(key, default_val):
    yaml_path = r"./config.yml"
    with open(yaml_path, "r") as f:
        datas = yaml.load(f, Loader=yaml.FullLoader)
    if key in datas:
        val = datas[key]
    else:
        val = default_val
    return val

def traversal_dir(dir_path):
    files_path = list()
    for root, dirs, files in os.walk(dir_path):
        #print("----------------------------------------")
        #print("root: ", root)
        #print("dirs: ", dirs)
        #print("files: ", len(files))
        #print("----------------------------------------")
        for file in files:
            if file.endswith(".py"):
                files_path.append(os.path.join(root, file))
    return files_path
def test_traversal_dir(dir_path):
    py_paths = traversal_dir(".")
    pattern = re.compile("inter_sec", re.M | re.I)
    for py_path in py_paths:
        # open the file to get the contnt
        with open(py_path, "r") as f:
            content = f.read()
        # use the regular module to find 
        ret = re.search(pattern, content)
        if ret:
            print(py_path)

if __name__ == "__main__":
    key = "backbone_normal"
    default_val = "hello, the world"
    val = get_config(key, default_val)
    print(val)
    print("hello, the world!")
        
        