import re


def get_sel_ids(sel):
    ids = []
    for field in sel.split(","):
        if re.match("^\d+$", field):
            ids.append(int(field))
        elif re.match("^\d+-\d+$", field):
            _min, _max = field.split("-")
            _min = int(_min)
            _max = int(_max)
            if _min >= _max:
                raise ValueError("Min id can not be lower than max id: %s" % field)
            ids.extend(list(range(_min, _max+1)))
        else:
            raise ValueError("Invalid entry ids selection : %s" % field)
    return ids


allowed_modifications = {"SEP": "S", "TPO": "T"}
modifications_substitutions = {"SEP": "D", "TPO": "D"}

def get_mod_seq(seq, modifications):
    if modifications:
        mod_data = {}
        for mod in modifications:
            if mod["code"] not in allowed_modifications:
                raise KeyError(mod["code"])
            sel_res = seq[mod["res_id"]]
            mod_res = allowed_modifications[mod["code"]]
            if sel_res != mod_res:
                raise ValueError(
                    f"Residue {sel_res} can not be modified into {mod_res}")
            mod_data[mod["res_id"]] = modifications_substitutions[mod["code"]]
        mod_seq = []
        for i,r in enumerate(seq):
            if i not in mod_data:
                mod_seq.append(r)
            else:
                mod_seq.append(mod_data[i])
        mod_seq = "".join(mod_seq)
        return mod_seq
    else:
        return seq