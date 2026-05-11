from torch.utils.data import DataLoader, Dataset
import copy
import torch
import torch.optim
import torch.nn.functional as F
from options import args_parser
from networks.models import DenseNet121
from utils import losses, ramps
from utils.util import get_timestamp
from semantic_anchor import anchor_calibration_loss
args = args_parser()

def get_current_consistency_weight(epoch):
     return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def update_ema_variables(model, ema_model, alpha, global_step):
     alpha = min(1 - 1 / (global_step + 1), alpha)
     for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha =  1-alpha)


class DatasetSplit(Dataset):
     def __init__(self, dataset, idxs):
          self.dataset = dataset
          self.idxs = list(idxs)

     def __len__(self):
        
          return len(self.idxs)

     def __getitem__(self, item):
     
          items, index, image,  label = self.dataset[self.idxs[item]]
          return items, index, image, label

class UnsupervisedLocalUpdate(object):
     def __init__(self, args, dataset=None, idxs=None):
          self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size = args.batch_size, shuffle = True)
          net = DenseNet121(out_size = 5, mode=args.label_uncertainty, drop_rate=args.drop_rate)
          if len(args.gpu.split(',')) > 1:
               net = torch.nn.DataParallel(net,device_ids=[0,1])
          self.ema_model = net.cuda()
          for param in self.ema_model.parameters():
               param.detach_()
          self.epoch = 0
          self.iter_num = 0
          self.flag = True
          self.base_lr = 2e-4
       
     def train(self, args, net, op_dict, epoch, semantic_anchors):
          net.train()
          self.optimizer = torch.optim.Adam(net.parameters(), lr=args.base_lr, betas=(0.9, 0.999), weight_decay=5e-4)
          self.optimizer.load_state_dict(op_dict)
          
          for param_group in self.optimizer.param_groups:
               param_group['lr'] = self.base_lr
          
          self.epoch = epoch
          if self.flag:
               self.ema_model.load_state_dict(net.state_dict())
               self.flag = False 
               print('done')
     
          epoch_loss = []
          print('begin training')
          for epoch in range(args.local_ep):

               batch_loss = []
               iter_max = len(self.ldr_train)

               for i, (_,_, (image_batch, ema_image_batch), label_batch) in enumerate(self.ldr_train):
                    

                    image_batch, ema_image_batch, label_batch = image_batch.cuda(), ema_image_batch.cuda(), label_batch.cuda()
                   
                    ema_inputs = ema_image_batch 
                    inputs = image_batch 

                    features, outputs = net(inputs)
                    
                    with torch.no_grad():
                         ema_activations, ema_output = self.ema_model(ema_inputs)
                    T = 10
                     
                    with torch.no_grad():
                         _, logits_sum = net(inputs)
                         for i in range(T):
                              _, logits = net(inputs)
                              logits_sum = logits_sum + logits
                         logits = logits_sum / (T+1) 
                         preds = F.softmax(logits, dim=1)
                         uncertainty = -1.0*torch.sum(preds*torch.log(preds + 1e-6), dim=1)
                         uncertainty_mask = (uncertainty < 2.0) 
                    
                    with torch.no_grad():
                         probs = F.softmax(outputs, dim=1)
                         confidence,_ = torch.max(probs, dim=1)
                         confidence_mask = (confidence>=0.3)
                    mask = confidence_mask * uncertainty_mask

                    anchor_scores = torch.matmul(
                         F.normalize(features.detach(), dim=1),
                         F.normalize(semantic_anchors.detach(), dim=1).t()
                    )
                    combined_logits = outputs + args.anchor_logit_weight * anchor_scores
                    combined_probs = F.softmax(combined_logits, dim=1)
                    combined_confidence, pseudo_idx = torch.max(combined_probs, dim=1)
                    combined_mask = combined_confidence >= args.pseudo_threshold
                    mask = mask & combined_mask

                    if mask.any():
                         pseudo_loss = F.cross_entropy(outputs[mask], pseudo_idx[mask].long())
                         anchor_align_loss = F.mse_loss(
                              F.normalize(features[mask], dim=1),
                              F.normalize(semantic_anchors[pseudo_idx[mask].long()].detach(), dim=1)
                         )
                    else:
                         pseudo_loss = torch.tensor(0.0, device=inputs.device)
                         anchor_align_loss = torch.tensor(0.0, device=inputs.device)

                   
                    
                    consistency_weight = get_current_consistency_weight(self.epoch)
                    consistency_dist = torch.sum(losses.softmax_mse_loss(outputs, ema_output)) / args.batch_size 
                    consistency_loss = consistency_dist
                   
                    calib_loss = anchor_calibration_loss(net, semantic_anchors.detach())
                    loss = (
                         15 * consistency_weight * consistency_loss
                         + args.anchor_pseudo_weight * pseudo_loss
                         + args.anchor_align_weight * anchor_align_loss
                         + args.anchor_calib_weight * calib_loss
                    )
                   
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    update_ema_variables(net, self.ema_model, args.ema_decay, self.iter_num)
                    batch_loss.append(loss.item())
                    
                    self.iter_num = self.iter_num + 1
                    
               timestamp = get_timestamp()

               
              
               self.epoch = self.epoch + 1
               epoch_loss.append(sum(batch_loss) / len(batch_loss))
               print(epoch_loss)
               
          return net.state_dict(), sum(epoch_loss) / len(epoch_loss), copy.deepcopy(self.optimizer.state_dict())  
