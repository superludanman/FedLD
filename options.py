import argparse
from networks.models import DenseNet121
def args_parser():
     parser = argparse.ArgumentParser()
     parser.add_argument('--root_path', type=str, default='./datasets/rsna/output/stage_2_train', help='dataset root dir')
     parser.add_argument('--csv_file_train', type=str, default='./datasets/training.csv', help='training set csv file')
     parser.add_argument('--csv_file_val', type=str, default='./datasets/validation.csv', help='validation set csv file')
     parser.add_argument('--csv_file_test', type=str, default='./datasets/testing.csv', help='testing set csv file')
     parser.add_argument('--batch_size', type=int, default=48, help='batch_size per gpu')
     parser.add_argument('--drop_rate', type=int, default=0.2, help='dropout rate')
     parser.add_argument('--ema_consistency', type=int, default=1, help='whether train baseline model')
     parser.add_argument('--base_lr', type=float,  default=2e-4, help='maximum epoch number to train')
     parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
     parser.add_argument('--seed', type=int,  default=1337, help='random seed')
     parser.add_argument('--gpu', type=str,  default='0,1', help='GPU to use')
     parser.add_argument('--local_ep', type=int,  default=1, help='local epoch')
     parser.add_argument('--num_users', type=int,  default=10, help='local epoch')
     parser.add_argument('--rounds', type=int,  default=200, help='local epoch')
     parser.add_argument('--num_classes', type=int, default=5, help='number of classes')


     ### tune
     parser.add_argument('--resume', type=str,  default=None, help='model to resume')
     parser.add_argument('--start_epoch', type=int,  default=0, help='start_epoch')
     parser.add_argument('--global_step', type=int,  default=0, help='global_step')
     ### costs
     parser.add_argument('--label_uncertainty', type=str,  default='U-Ones', help='label type')
     parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
     parser.add_argument('--consistency', type=float,  default=1, help='consistency')
     parser.add_argument('--consistency_rampup', type=float,  default=30, help='consistency_rampup') 
     parser.add_argument('--anchor_momentum', type=float, default=0.9, help='EMA momentum for semantic anchors')
     parser.add_argument('--anchor_proto_weight', type=float, default=1.0, help='weight for prototype-anchor regularization')
     parser.add_argument('--anchor_calib_weight', type=float, default=0.1, help='weight for anchor calibration loss')
     parser.add_argument('--anchor_align_weight', type=float, default=1.0, help='weight for unlabeled anchor alignment loss')
     parser.add_argument('--anchor_pseudo_weight', type=float, default=1.0, help='weight for anchor-guided pseudo label loss')
     parser.add_argument('--anchor_logit_weight', type=float, default=0.5, help='weight for anchor logits when building pseudo labels')
     parser.add_argument('--pseudo_threshold', type=float, default=0.7, help='confidence threshold for pseudo labels')
     args = parser.parse_args()
     return args
