import torch
import torch.nn as nn

from tools.utils.misc import get_rank

class BaseModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        # self.config = config.model.geometry
        self.config = config
        # self.config_full = config
        self.rank = get_rank()
        self.setup()
        # import pdb
        # pdb.set_trace()
        # if self.config.get('weights', None):
        #     self.load_state_dict(torch.load(self.config.weights), strict = False)
    
    def setup(self):
        raise NotImplementedError
    
    def update_step(self, epoch, global_step):
        pass
    
    def train(self, mode=True):
        return super().train(mode=mode)
    
    def eval(self):
        return super().eval()
    
    def regularizations(self, out):
        return {}
    
    @torch.no_grad()
    def export(self, export_config):
        return {}
