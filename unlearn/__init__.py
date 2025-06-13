from .fisher import fisher, fisher_new
from .FT import FT, FT_l1

from .impl import load_unlearn_checkpoint, save_unlearn_checkpoint
from .retrain import retrain

from .Wfisher import Wfisher

from .NGPlus import NGPlus
from .SRL import SRL

from .SalUn import SalUn

from .ideal import ideal

from .vargrad import VarGrad
from .salgrad import SalGrad
from .espgrad import EspGrad


def raw(data_loaders, model, criterion, args):
    pass


def get_unlearn_method(name):
    """method usage:

    function(data_loaders, model, criterion, args)"""
    if name == "raw":
        return raw
    elif name == "GA":
        return GA
    elif name == "FT":
        return FT
    elif name == "FT_l1":
        return FT_l1
    elif name == "fisher":
        return fisher
    elif name == "retrain":
        return retrain
    elif name == "fisher_new":
        return fisher_new
    elif name == "wfisher":
        return Wfisher
    elif name == "GA_l1":
        return GA_l1
    elif name == "retrain_ls":
        return retrain_ls
    elif name == "retrain_sam":
        return retrain_sam
    elif name == "mask_NGPlus":
        return mask_NGPlus
    elif name == "NGPlus":
        return NGPlus
    elif name == "mask_SRL":
        return mask_SRL
    elif name == "SRL":
        return SRL
    elif name == "barrier":
        return barrier
    elif name == "mask_barrier":
        return mask_barrier
    elif name == "SalUn":
        return SalUn
    elif name == "mix_NGPlus":
        return mix_NGPlus
    elif name == "mix_SRL":
        return mix_SRL
    elif name == "ideal":
        return ideal
    elif name == "nothing":
        return nothing
    elif name == "pSalUn":
        return pSalUn
    elif name == "VarGrad":
        return VarGrad
    elif name == "SalGrad":
        return SalGrad
    elif name == "EspGrad":
        return EspGrad
    elif name == "ProbGrad":
        return ProbGrad
    elif name == "FocalGrad":
        return FocalGrad
    else:
        raise NotImplementedError(f"Unlearn method {name} not implemented!")
