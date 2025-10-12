import os

def link_file(src, target):
    if os.path.isdir(target) or os.path.isfile(target):
        os.system('rm -rf {}'.format(target))
    os.system('ln -s {} {}'.format(src, target))