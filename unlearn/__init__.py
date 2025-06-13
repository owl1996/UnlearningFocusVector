from .impl import load_unlearn_checkpoint, save_unlearn_checkpoint

from .ideal import ideal
from .retrain import retrain
from .FT import FT

from .fisher import fisher, fisher_new
from .Wfisher import Wfisher

from .NGPlus import NGPlus
from .NGradMask import NGradMask
from .NGSalUn import NGSalUn
from .NGradFocus import NGradFocus

from .SRL import SRL
from .SRGradFocus import SRGradFocus
from .SRGradMask import SRGradMask
from .SalUn import SalUn

def get_unlearn_method(name):
    """method usage:

    function(data_loaders, model, criterion, args)"""
    if name == "ideal":
        return ideal
    elif name == "retrain":
        return retrain
    elif name == "FT":
        return FT

    elif name == "fisher":
        return fisher
    elif name == "fisher_new":
        return fisher_new
    elif name == "wfisher":
        return Wfisher
    
    elif name == "NGPlus":
        return NGPlus
    elif name == "NGradMask":
        return NGradMask
    elif name == "NGradFocus":
        return NGradFocus
    elif name == "NGSalUn":
        return NGSalUn
    
    elif name == "SRL":
        return SRL
    elif name == "SRGradFocus":
        return SRGradFocus
    elif name == "SRGradMask":
        return SRGradMask
    elif name == "SalUn":
        return SalUn
    
    else:
        raise NotImplementedError(f"Unlearn method {name} not implemented!")
