from rdkit import Chem
import numpy as np

def make_smile_canonical(smile): 

    '''
    To avoid duplicates, for example: canonical '*C=C(*)C' == '*C(=C*)C'
    '''
    try:
        mol = Chem.MolFromSmiles(smile)
        canon_smile = Chem.MolToSmiles(mol, canonical=True)
        return canon_smile
    except:
        return np.nan