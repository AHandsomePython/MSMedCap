import torch
from collections import OrderedDict



def merge_dict(samblippath, mlppath, savepath):
    samblipdict =  torch.load(samblippath, map_location = "cpu")['model_state_dict']
    mlpdict =  torch.load(mlppath, map_location = "cpu")['model_state_dict']
    dict0 = OrderedDict()
    for key in samblipdict.keys():
        if key == 'opt_model.model.decoder.embed_tokens.weight':
            break
        dict0[key] = samblipdict[key]
    
    for key in mlpdict.keys():
        if key not in dict0.keys():
            dict0[key] = mlpdict[key]
    
    torch.save(dict0, savepath)