import torch


def named_params(curr_module, prefix=''): 
    memo=set()
    if hasattr(curr_module, 'named_leaves'):
        for name, p in curr_module.named_leaves():
            if p is not None and p not in memo:
                memo.add(p)
                yield prefix + ('.' if prefix else '') + name, p

    for mname, module in curr_module.named_children():
            submodule_prefix = prefix + ('.' if prefix else '') + mname
            for name, p in named_params(module, submodule_prefix):
                yield name, p

def set_param(curr_mod, name, param):
    if '.' in name:
        n = name.split('.')
        module_name = n[0]
        rest = '.'.join(n[1:])
        for name, mod in curr_mod.named_children():
            if module_name == name:
                set_param(mod, rest, param)
                break
    else:
        setattr(curr_mod, name, param)

def inner_update(curr_mod,lr):
    for name, p  in  named_params(curr_mod):
        set_param(curr_mod, name+"_meta", p-lr*p.grad)

def outer_update(curr_mod,lr):
    with torch.no_grad():
        max_grad=0
        for _, param in curr_mod.named_parameters():
            max_grad=max(torch.max(torch.abs(param.grad)).item(),max_grad)
        if max_grad >0:  
            for _, param in curr_mod.named_parameters():
                param.add_(-lr*(param.grad/max_grad))