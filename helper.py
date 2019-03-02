import codecs
import os
import json
import numpy as np
def write_numpy_to_json(array,file_name):
    if not os.path.exists(os.path.join(os.getcwd(),'matches')):
        os.makedirs(os.path.join(os.getcwd(),'matches'))
    file_path = os.path.join(os.getcwd(),'matches',file_name+".json")
    array  = array.tolist()
    json.dump(array, codecs.open(file_path, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)

def convert_json_to_numpy(file_name):

    file_path = os.path.join(os.getcwd(),'matches',file_name+".json")
    obj_text = codecs.open(file_path, 'r', encoding='utf-8').read()
    b_new = json.loads(obj_text)
    np_array = np.array(b_new)
    return  np_array