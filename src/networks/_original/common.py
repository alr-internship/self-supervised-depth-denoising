import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

def set_param_grad(model, value, set_eval_mode=True):
    for param in model.parameters():
        param.requires_grad = value
    
    if set_eval_mode:
        model.eval()


class MultiHead(nn.Module):
    def __init__(self, in_features=None, num_classes=None):
        super(MultiHead, self).__init__()

        heads = [torch.nn.Linear(in_features, x) for x in num_classes]
        self.heads = torch.nn.ModuleList(heads)

    def __len__(self):
        return len(self._modules)

    def forward(self, input):
        return tuple([head(input) for head in self.heads])

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def add_module(self, module):
    self.add_module(str(len(self) + 1), module)
    
torch.nn.Module.add = add_module

class Concat(nn.Module):
    def __init__(self, dim, *args):
        super(Concat, self).__init__()
        self.dim = dim

        for idx, module in enumerate(args):
            self.add_module(str(idx), module)

    def forward(self, input):
        inputs = []
        for module in self._modules.values():
            inputs.append(module(input))

        inputs_shapes2 = [x.shape[2] for x in inputs]
        inputs_shapes3 = [x.shape[3] for x in inputs]        

        if np.all(np.array(inputs_shapes2) == min(inputs_shapes2)) and np.all(np.array(inputs_shapes3) == min(inputs_shapes3)):
            inputs_ = inputs
        else:
            target_shape2 = min(inputs_shapes2)
            target_shape3 = min(inputs_shapes3)

            inputs_ = []
            for inp in inputs: 
                diff2 = (inp.size(2) - target_shape2) // 2 
                diff3 = (inp.size(3) - target_shape3) // 2 
                inputs_.append(inp[:, :, diff2: diff2 + target_shape2, diff3:diff3 + target_shape3])

        return torch.cat(inputs_, dim=self.dim)

    def __len__(self):
        return len(self._modules)


class Swish(nn.Module):
    """
        https://arxiv.org/abs/1710.05941
        The hype was so huge that I could not help but try it
    """
    def __init__(self):
        super(Swish, self).__init__()
        self.s = nn.Sigmoid()

    def forward(self, x):
        return x * self.s(x)



def act(act_fun = 'LeakyReLU'):
    '''
        Either string defining an activation function or module (e.g. nn.ReLU)
    '''
    if isinstance(act_fun, str):
        if act_fun == 'LeakyReLU':
            return nn.LeakyReLU(0.2, inplace=True)
        elif act_fun == 'Swish':
            return Swish()
        elif act_fun == 'ELU':
            return nn.ELU()
        elif act_fun == 'none':
            return nn.Sequential()
        else:
            assert False
    else:
        return act_fun()


def norm(num_features, tp='bn', **kwargs):
    if tp == 'bn':
        return nn.BatchNorm2d(num_features, **kwargs)
    elif tp == 'in':
        return nn.InstanceNorm2d(num_features, **kwargs)
    elif tp == 'none':
        return Identity()


def conv(in_f, out_f, kernel_size, stride=1, bias=True, pad='zero', downsample_mode='stride', conv_class=nn.Conv2d):
    downsampler = None
    if stride != 1 and downsample_mode != 'stride':

        if downsample_mode == 'avg':
            downsampler = nn.AvgPool2d(stride, stride)
        elif downsample_mode == 'max':
            downsampler = nn.MaxPool2d(stride, stride)
        elif downsample_mode  in ['lanczos2', 'lanczos3']:
            downsampler = Downsampler(n_planes=out_f, factor=stride, kernel_type=downsample_mode, phase=0.5, preserve_size=True)
        else:
            assert False

        stride = 1

    padder = None
    to_pad = int((kernel_size - 1) / 2)
    if pad == 'reflection' and to_pad != 0:
        padder = nn.ReflectionPad2d(to_pad)
        to_pad = 0
  
    convolver = conv_class(in_f, out_f, kernel_size, stride, padding=to_pad, bias=bias)

    # print (layers22)
    layers = filter(lambda x: x is not None, [padder, convolver, downsampler])
    # print (layers)
    return nn.Sequential(*layers)


class ListModule(nn.Module):
    def __init__(self, *args):
        super(ListModule, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        if idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        if idx < 0: 
            idx = len(self) + idx

        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)



class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, sz=None):
        super().__init__()
        sz = sz or (1,1)
        self.ap = nn.AdaptiveAvgPool2d(sz)
        self.mp = nn.AdaptiveMaxPool2d(sz)

    def forward(self, x): 
        return torch.cat([self.mp(x), self.ap(x)], 1)




class BaseModel(torch.nn.Module):
    def __init__(self, feature_extractor, predictor, pooling='avg'):
        super(BaseModel, self).__init__()
        
        self.feature_extractor = feature_extractor
        self.predictor = predictor

        if pooling == 'avg':
            self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        elif pooling == 'max':
            self.pool = torch.nn.AdaptiveMaxPool2d((1, 1))
        elif pooling == 'concat':
            self.pool = torch.nn.AdaptiveConcatPool2d((1, 1))
        else:
            raise False


    def forward(self, input):
        x = self.feature_extractor(input)

        x = self.pool(x)
        x = x.view(x.size(0), -1)
        
        x = self.predictor(x)

        return x

class BloorPool(nn.Module):
    def __init__(self, pad_type='zero', filt_size=3, stride=2, channels=None, pad_off=0):
        super(BloorPool, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2)), int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2))]
        self.pad_sizes = [pad_size+pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride-1)/2.)
        self.channels = channels

        # print('Filter size [%i]'%filt_size)
        if(self.filt_size==1):
            a = np.array([1.,])
        elif(self.filt_size==2):
            a = np.array([1., 1.])
        elif(self.filt_size==3):
            a = np.array([1., 2., 1.])
        elif(self.filt_size==4):    
            a = np.array([1., 3., 3., 1.])
        elif(self.filt_size==5):    
            a = np.array([1., 4., 6., 4., 1.])
        elif(self.filt_size==6):    
            a = np.array([1., 5., 10., 10., 5., 1.])
        elif(self.filt_size==7):    
            a = np.array([1., 6., 15., 20., 15., 6., 1.])

        filt = torch.Tensor(a[:,None]*a[None,:])
        filt = filt/torch.sum(filt)
        self.register_buffer('filt', filt[None,None,:,:].repeat((self.channels,1,1,1)))

        self.pad = self.get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if(self.filt_size==1):
            if(self.pad_off==0):
                return inp[:,:,::self.stride,::self.stride]    
            else:
                return self.pad(inp)[:,:,::self.stride,::self.stride]
        else:
            # print("Input: ", inp.shape)
            # print("Filter: ", self.filt.shape)
            return F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])

    def get_pad_layer(self, pad_type):
        if(pad_type in ['refl','reflect']):
            PadLayer = nn.ReflectionPad2d
        elif(pad_type in ['repl','replicate']):
            PadLayer = nn.ReplicationPad2d
        elif(pad_type=='zero'):
            PadLayer = nn.ZeroPad2d
        else:
            print('Pad type [%s] not recognized'%pad_type)
        return PadLayer

     
### https://github.com/automan000/Convolution_LSTM_PyTorch/blob/master/convolution_lstm.py
from torch.autograd import Variable
class ConvLSTMCell(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvLSTMCell, self).__init__()

        assert hidden_channels % 2 == 0

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_features = 4

        self.padding = int((kernel_size - 1) / 2)

        self.Wxi = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whi = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxf = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whf = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxc = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whc = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxo = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Who = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)

        self.Wci = None
        self.Wcf = None
        self.Wco = None

    def forward(self, x, h, c):
        ci = torch.sigmoid(self.Wxi(x) + self.Whi(h) + c * self.Wci)
        cf = torch.sigmoid(self.Wxf(x) + self.Whf(h) + c * self.Wcf)
        cc = cf * c + ci * torch.tanh(self.Wxc(x) + self.Whc(h))
        co = torch.sigmoid(self.Wxo(x) + self.Who(h) + cc * self.Wco)
        ch = co * torch.tanh(cc)
        return ch, cc

    def init_hidden(self, batch_size, shape):
        if self.Wci is None:
            self.Wci = Variable(torch.zeros(1, self.hidden_channels, shape[0], shape[1])).cuda()
            self.Wcf = Variable(torch.zeros(1, self.hidden_channels, shape[0], shape[1])).cuda()
            self.Wco = Variable(torch.zeros(1, self.hidden_channels, shape[0], shape[1])).cuda()
        else:
            assert shape[0] == self.Wci.size()[2], 'Input Height Mismatched!'
            assert shape[1] == self.Wci.size()[3], 'Input Width Mismatched!'
        return (Variable(torch.zeros(batch_size, self.hidden_channels, shape[0], shape[1])).cuda(),
                Variable(torch.zeros(batch_size, self.hidden_channels, shape[0], shape[1])).cuda())