

def flatten_json(jsn):
    out = {}

    def flatten(x, name=''):
        if type(x) is dict:
            for a in x:
                flatten(x[a], name + a + '_')
        elif type(x) is list:
            i = 0
            for a in x:
                flatten(a, name + str(i) + '_')
                i += 1
        else:
            out[name[:-1]] = x

    flatten(jsn)
    return out


def flatten_dict_list(lst):
    return {k: v for d in lst for k, v in d.items()}


def rekey(inp_dict, keys_replace):
    return {keys_replace.get(k, k): v for k, v in inp_dict.items()}


def list_of_dicts_to_dict(dictlist):
    return dict((key, val) for k in dictlist for key, val in k.items())