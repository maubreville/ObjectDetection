from fastai import *
from fastai.vision import *
from fastai.callbacks import *
from fastai.vision.models.unet import _get_sfs_idxs

# export
class LateralUpsampleMerge(nn.Module):

    def __init__(self, ch, ch_lat, hook):
        super().__init__()
        self.hook = hook
        self.conv_lat = conv2d(ch_lat, ch, ks=1, bias=True)

    def forward(self, x):
        return self.conv_lat(self.hook.stored) + F.interpolate(x, scale_factor=2)

def activ_to_bbox(acts, anchors, flatten=True):
    "Extrapolate bounding boxes on anchors from the model activations."
    if flatten:
        acts = acts * acts.new_tensor([[0.1, 0.1, 0.2]])
        centers = anchors[...,2:] * acts[...,:2] + anchors[...,:2]
        sizes = anchors[...,2:] * torch.exp(acts[...,2:])
        return torch.cat([centers, sizes], -1)
    else: return [activ_to_bbox(act,anc) for act,anc in zip(acts, anchors)]
    return res

class STNClassifier(nn.Module):
    "Implements RetinaNet from https://arxiv.org/abs/1708.02002"

    def __init__(self, encoder: nn.Module, n_classes, final_bias:float=0.,  n_conv:float=4,
                 chs=256, n_anchors=9, anchors=None,  oversize=1.5, flatten=True, sizes=None, patchClassifier:nn.Module=None):
        super().__init__()
        self.n_classes, self.flatten = n_classes, flatten
        imsize = (256, 256)
        self.sizes = sizes
        self.grid_target_size = (64,64)
        self._sizes = torch.Tensor(sizes)
        self._oversize = oversize
        
        sfs_szs, x, hooks = self._model_sizes(encoder, size=imsize)
        sfs_idxs = _get_sfs_idxs(sfs_szs)
        self.encoder = encoder
        self.c5top5 = conv2d(sfs_szs[-1][1], chs, ks=1, bias=True)
        self.c5top6 = conv2d(sfs_szs[-1][1], chs, stride=2, bias=True)
        self.p6top7 = nn.Sequential(nn.ReLU(), conv2d(chs, chs, stride=2, bias=True))
        self.merges = nn.ModuleList([LateralUpsampleMerge(chs, szs[1], hook)
                                     for szs, hook in zip(sfs_szs[-2:-4:-1], hooks[-2:-4:-1])])
        self.smoothers = nn.ModuleList([conv2d(chs, chs, 3, bias=True) for _ in range(3)])
        self.classifier = self._head_subnet(n_classes, n_anchors, final_bias, chs=chs, n_conv=n_conv)
        self.box_regressor = self._head_subnet(3, n_anchors, 0., chs=chs, n_conv=n_conv)
        if (patchClassifier is None):
            self.box_stn = create_cnn_model(models.resnet18, n_classes, cut=None, pretrained=True, lin_ftrs=None, ps=0.5,   split_on=None, bn_final=False, concat_pool=True)
            
#            self._stn_subnet(n_classes, final_bias, chs=chs, n_conv=n_conv)
        else:
            self.box_stn = patchClassifier
        self.anchors = anchors

    def _head_subnet(self, n_classes, n_anchors, final_bias=0., n_conv=4, chs=256):
        layers = [self._conv2d_relu(chs, chs, bias=True) for _ in range(n_conv)]
        layers += [conv2d(chs, n_classes * n_anchors, bias=True)]
        layers[-1].bias.data.zero_().add_(final_bias)
        layers[-1].weight.data.fill_(0)
        return nn.Sequential(*layers)
   
    def _stn_subnet(self, n_classes, final_bias=0., chs=256, n_conv=4):
        self.stn_cla_body = create_body(models.resnet18, True, -2)
        
        layers = [self._conv2d_relu(512, 128, bias=True, stride=2)]
        layers += [conv2d(128, self.n_classes, bias=True)]

        
        self.stn_cla_head = nn.Sequential(*layers) #self._head_subnet(n_classes, 1, final_bias, chs=512, n_conv=n_conv)

    def _apply_transpose(self, func, p_states, n_classes):
        if not self.flatten:
            sizes = [[p.size(0), p.size(2), p.size(3)] for p in p_states]
            return [func(p).permute(0, 2, 3, 1).view(*sz, -1, n_classes) for p, sz in zip(p_states, sizes)]
        else:
            return torch.cat(
                [func(p).permute(0, 2, 3, 1).contiguous().view(p.size(0), -1, n_classes) for p in p_states], 1)

    def _model_sizes(self, m: nn.Module, size:tuple=(256,256), full:bool=True) -> Tuple[Sizes,Tensor,Hooks]:
        "Passes a dummy input through the model to get the various sizes"
        hooks = hook_outputs(m)
        ch_in = in_channels(m)
        x = torch.zeros(1,ch_in,*size)
        x = m.eval()(x)
        res = [o.stored.shape for o in hooks]
        if not full: hooks.remove()
        return res,x,hooks if full else res

    def _conv2d_relu(self, ni:int, nf:int, ks:int=3, stride:int=1,
                    padding:int=None, bn:bool=False, bias=True) -> nn.Sequential:
        "Create a `conv2d` layer with `nn.ReLU` activation and optional(`bn`) `nn.BatchNorm2d`"
        layers = [conv2d(ni, nf, ks=ks, stride=stride, padding=padding, bias=bias), nn.ReLU()]
        if bn: layers.append(nn.BatchNorm2d(nf))
        return nn.Sequential(*layers)
    
    
    def grid_sample(self, x, reg_out):
        #print('CLA OUT: ',cla_out.shape)
        #print('REG OUT: ',reg_out.shape)
        #print('Anchors:',self.anchors.shape)
        bbox_pred = activ_to_bbox(reg_out, self.anchors.to(reg_out.device))
        #print('BBOX: ',bbox_pred.shape)
        bbox_theta = torch.cat((bbox_pred[...,2:3]*self._oversize,torch.zeros_like(bbox_pred[...,0:1]),
                        bbox_pred[...,1:2],torch.zeros_like(bbox_pred[...,0:1]),
                        bbox_pred[...,2:3]*self._oversize,bbox_pred[...,0:1] ), 2)
        #print('BBOX theta:', bbox_theta.shape)
        theta = bbox_theta.view(-1, 2, 3)
        #print('THETA: ',theta.shape) 
        tsize = (int(x.shape[0]*sum(self._sizes**2)), *x.shape[1:])
        #print('Target Expansion:', tsize, (-1, *x.shape[1:]))
        
        if (x.shape[0]==1):
            patchRep = x.repeat((int(sum(self._sizes**2)),1,1,1))
        else:
#            patchRep = x[None,:,:,:,:].expand((int(sum(self._sizes**2)),*x.shape[0:]))
            patchRep = x[None,:,:,:,:].repeat((int(sum(self._sizes**2)),1,1,1,1)).permute(1,0,2,3,4)

            #print('PatchRep:', patchRep.shape)
            patchRep = patchRep.contiguous().view(-1, *x.shape[1:])
#            patchRep = patchRep.view(-1, *x.shape[1:])
        #print('PatchRep: ', patchRep.shape)
        grid = torch.nn.functional.affine_grid(theta, (tsize[0],3,*self.grid_target_size))
        #print('Grid: ', grid.shape)
        StnOut = torch.nn.functional.grid_sample(patchRep, grid)

        return StnOut, theta


    def predict_gridtiles(self, x):
        c5 = self.encoder(x)
        p_states = [self.c5top5(c5.clone()), self.c5top6(c5)]
        p_states.append(self.p6top7(p_states[-1]))
        #print('Called forward!')
        for merge in self.merges:
            p_states = [merge(p_states[0])] + p_states
        for i, smooth in enumerate(self.smoothers[:3]):
            p_states[i] = smooth(p_states[i])
        if self.sizes is not None:
            p_states = [p_state for p_state in p_states if p_state.size()[-1] in self.sizes]
        #print('Self.classifier',self.classifier,p_states,self.n_classes)
        #print('Self.bbox_regressor',self.box_regressor, p_states, 4)
        cla_out = self._apply_transpose(self.classifier, p_states, self.n_classes)
        reg_out = self._apply_transpose(self.box_regressor, p_states, 3)
        
        reg_out_mod = reg_out.clone()
#        StnOut,theta = self.grid_sample(x, reg_out)
        reg_out_mod[cla_out[...,0]<0.1] = 0
        reg_out_mod = reg_out_mod.sigmoid()
        
        StnOut,theta = self.grid_sample(x, reg_out_mod)

        
        
        return StnOut, theta
    
    def forward(self, x):
        c5 = self.encoder(x)
        p_states = [self.c5top5(c5.clone()), self.c5top6(c5)]
        p_states.append(self.p6top7(p_states[-1]))
        #print('Called forward!')
        for merge in self.merges:
            p_states = [merge(p_states[0])] + p_states
        for i, smooth in enumerate(self.smoothers[:3]):
            p_states[i] = smooth(p_states[i])
        if self.sizes is not None:
            p_states = [p_state for p_state in p_states if p_state.size()[-1] in self.sizes]
        #print('Self.classifier',self.classifier,p_states,self.n_classes)
        #print('Self.bbox_regressor',self.box_regressor, p_states, 4)
        cla_out = self._apply_transpose(self.classifier, p_states, self.n_classes)
        reg_out = self._apply_transpose(self.box_regressor, p_states, 3)
        
        reg_out_mod = reg_out.clone()

        # zero-out all non-active regions to not confuse the classifier too much
        reg_out_mod[cla_out[...,0]<0.1] = 0
        reg_out_mod = reg_out_mod.sigmoid()
        
        StnOut,theta = self.grid_sample(x, reg_out_mod)
        #print('Out:', StnOut.shape)
        claOut = self.box_stn(StnOut)
        
        #print('claStem:',claStem.shape)
        #claOut = self.stn_cla_head(claStem)[...,0,0]
        #print('claOut', claOut.shape)
        claOut = claOut.view((-1, int(sum(self._sizes**2)), self.n_classes))
        #print('claOut', claOut.shape)


        
        return [cla_out,
                reg_out,
                claOut,
                [[p.size(2), p.size(3)] for p in p_states]]


