import yaml

def get_config(key, default_val):
    yaml_path = r"./config.yml"
    with open(yaml_path, "r") as f:
        datas = yaml.load(f, Loader=yaml.FullLoader)
    if key in datas:
        val = datas[key]
    else:
        val = default_val
    return val

# test
if __name__ == "__main__":
    key = "backbone_normal"
    default_val = "hello, the world"
    val = get_config(key, default_val)
    print(val)