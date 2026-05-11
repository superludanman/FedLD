import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch
import torch.optim
from options import args_parser
import copy
from utils import losses
import torch.nn.functional as F
from semantic_anchor import compute_batch_prototypes, reduce_prototypes, anchor_calibration_loss

args = args_parser()


class DatasetSplit(Dataset):
     def __init__(self, dataset, idxs):
          self.dataset = dataset
          self.idxs = list(idxs)

     def __len__(self):
         
          return len(self.idxs)

     def __getitem__(self, item):
         
          items, index, image,  label = self.dataset[self.idxs[item]]
          return items, index, image, label

class SupervisedLocalUpdate(object):
     def __init__(self, args, dataset, idxs):
          
          self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size = args.batch_size, shuffle = True)
          
          self.epoch = 0
          self.iter_num = 0
          self.proto_sum = torch.zeros((args.num_classes, 1024)).cuda()
          self.proto_count = torch.zeros((args.num_classes,)).cuda()
          self.base_lr = args.base_lr
         
     def train(self, args, net, op_dict, semantic_anchors):
          net.train()
          self.optimizer = torch.optim.Adam(net.parameters(), lr=args.base_lr, betas=(0.9, 0.999), weight_decay=5e-4)
          self.optimizer.load_state_dict(op_dict)
          
          for param_group in self.optimizer.param_groups:
               param_group['lr'] = self.base_lr
          
          loss_fn = losses.LabelSmoothingCrossEntropy()
          self.proto_sum = torch.zeros((args.num_classes, semantic_anchors.size(1))).cuda()
          self.proto_count = torch.zeros((args.num_classes,)).cuda()
        # train and update
          epoch_loss = []
          print('begin training')
          for epoch in range(args.local_ep):
               batch_loss = []
               iter_max = len(self.ldr_train)
               print(iter_max) 
               for i, (_,_, (image_batch, ema_image_batch), label_batch) in enumerate(self.ldr_train):
                    image_batch, ema_image_batch, label_batch = image_batch.cuda(), ema_image_batch.cuda(), label_batch.cuda()
                    ema_inputs = ema_image_batch 
                    inputs = image_batch 
                    activations, outputs = net(inputs)
                    _, aug_outputs = net(ema_inputs)
                    loss_classification = loss_fn(outputs , label_batch.long()) + loss_fn(aug_outputs , label_batch.long())
                    batch_proto_sum, batch_proto_count = compute_batch_prototypes(activations, label_batch, args.num_classes)
                    with torch.no_grad():
                         self.proto_sum += batch_proto_sum.detach()
                         self.proto_count += batch_proto_count.detach()

                    batch_proto, batch_valid = reduce_prototypes(batch_proto_sum, batch_proto_count)
                    if batch_valid.any():
                         proto_reg_loss = F.mse_loss(
                              F.normalize(batch_proto[batch_valid], dim=1),
                              F.normalize(semantic_anchors[batch_valid], dim=1)
                         )
                    else:
                         proto_reg_loss = torch.tensor(0.0, device=activations.device)

                    calib_loss = anchor_calibration_loss(net, semantic_anchors.detach())
                    loss = loss_classification + args.anchor_proto_weight * proto_reg_loss + args.anchor_calib_weight * calib_loss
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    batch_loss.append(loss.item())
                    self.iter_num = self.iter_num + 1
               

               self.epoch = self.epoch + 1
               epoch_loss.append(np.array(batch_loss).mean())
               print(epoch_loss)
          return (
               net.state_dict(),
               sum(epoch_loss) / len(epoch_loss),
               copy.deepcopy(self.optimizer.state_dict()),
               self.proto_sum.detach().clone(),
               self.proto_count.detach().clone()
          )
