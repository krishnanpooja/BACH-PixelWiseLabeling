import sys
import torch
import visdom
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import matplotlib
matplotlib.use('Agg')
from torch.autograd import Variable
from torch.utils import data
from tqdm import tqdm
from ptsemseg.loader.BACHLoader import BACHLoader
import matplotlib.pyplot as plt
from ptsemseg.models import get_model
from ptsemseg.loader import get_loader, get_data_path
from ptsemseg.metrics import runningScore
from ptsemseg.loss import *
from ptsemseg.augmentations import *
import operator

SEG_LABELS_LIST = [
    {"id": 3, "name": "void",       "rgb_values": [0,   0,    0]},
    {"id": 0,  "name": "benign",   "rgb_values": [250, 0,    0]},
    {"id": 1,  "name": "in situ",      "rgb_values": [0,   150,  0]},
    {"id": 2,  "name": "invasive",       "rgb_values": [0, 0,  70]}]


def decode_segmap(label_img):
    label_img = np.squeeze(label_img)
    labels = np.unique(label_img)
    unique, counts = np.unique(label_img, return_counts=True)
    x=dict(zip(unique,counts))
    y=sorted(x.items(), key=operator.itemgetter(1))
    print('Unique counts of labels in decode_segmap:',y)

    label_infos = [l for l in SEG_LABELS_LIST if l['id'] in labels]

    label_img_rgb = np.array([label_img,
                              label_img,
                              label_img]).transpose(1,2,0)
    for l in label_infos:
        mask = label_img == l['id']
        label_img_rgb[mask] = l['rgb_values']

    return label_img_rgb.astype(np.uint8)



def train(args):

    if torch.cuda.is_available():
      print('Cuda is available')
    else:
      print('Cuda not available')
 # Setup Augmentations
    data_aug= Compose([RandomRotate(10),                                        
                       RandomHorizontallyFlip()])

    # Setup Dataloader
    data_loader = get_loader(args.dataset)
    data_path = get_data_path(args.dataset)
    t_loader = data_loader(data_path, is_transform=True, img_size=(args.img_rows, args.img_cols), augmentations=data_aug)
    v_loader = data_loader(data_path, is_transform=True, split='val', img_size=(args.img_rows, args.img_cols))

    n_classes = t_loader.n_classes
    print('Classes in train:',n_classes)
    trainloader = data.DataLoader(t_loader, batch_size=args.batch_size, num_workers=8, shuffle=True)
    valloader = data.DataLoader(v_loader, batch_size=args.batch_size, num_workers=8)

    # Setup Metrics
    running_metrics = runningScore(n_classes)
        
    # Setup visdom for visualization
    if args.visdom:
        vis = visdom.Visdom()

        loss_window = vis.line(X=torch.zeros((1,)).cpu(),
                           Y=torch.zeros((1)).cpu(),
                           opts=dict(xlabel='minibatches',
                                     ylabel='Loss',
                                     title='Training Loss'))

    # Setup Model
    model = get_model(args.arch, n_classes)
    
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    model.cuda()
    
    # Check if model has custom optimizer / loss
    optimizer = torch.optim.SGD(model.parameters(), lr=args.l_rate, momentum=0.99, weight_decay=5e-4)

    if hasattr(model.module, 'loss'):
        print('Using custom loss')
        loss_fn = model.module.loss
    else:
        loss_fn = cross_entropy2d

    if args.resume is not None:                                         
        if os.path.isfile(args.resume):
            print("Loading model and optimizer from checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            print("Loaded checkpoint '{}' (epoch {})"                    
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("No checkpoint found at '{}'".format(args.resume)) 

    best_iou = -100.0 
    for epoch in range(args.n_epoch):
        model.train()
        for j in range(10):
          for i, (images, labels) in enumerate(trainloader):
              images = Variable(images.cuda())
              labels = Variable(labels.cuda())

              optimizer.zero_grad()
              outputs = model(images)

              loss = loss_fn(input=outputs, target=labels)
            #loss = F.kl_div(input=outputs,target=labels.float()) 
              loss.backward()
              optimizer.step()

              if args.visdom:
                  vis.line(
                    X=torch.ones((1, 1)).cpu() * i,
                    Y=torch.Tensor([loss.data[0]]).unsqueeze(0).cpu(),
                    win=loss_window,
                    update='append')

            #if (i+1) % 20 == 0:
              print("Epoch [%d/%d] Loss: %.4f" % (epoch+1, args.n_epoch, loss.data[0]))

          model.eval()
        plt.figure(figsize=(15, 5 * 1))
        for i_val, (images_val, labels_val) in tqdm(enumerate(trainloader)):
            images_val = Variable(images_val.cuda(), volatile=True)
            labels_val = Variable(labels_val.cuda(), volatile=True)

            outputs = model(images_val)
            pred = outputs.data.max(1)[1].cpu().numpy()
            gt = labels_val.data.cpu().numpy()
            #img
            plt.subplot(5, 3,i*3+1)
	    plt.axis('off')
            images_val=images_val.squeeze(0)
	    #plt.imshow(images_val.data.cpu().numpy().transpose(1,2,0))
            #plt.savefig('/home/deepita/pytorch-semseg-master/input_img.png')
           
	    if i == 0:
  	      plt.title("Input image")
            #target
            plt.subplot(5, 3, i*3+2)
            plt.axis('off')
            plt.imshow(decode_segmap(gt))
            plt.savefig('/home/deepita/pytorch-semseg-master/target_img1.png')
            if i == 0:
              plt.title("Target image")

	    #pred
	    plt.subplot(5, 3, i * 3 + 3)
	    plt.axis('off')
	    plt.imshow(decode_segmap(pred))
            plt.savefig('/home/deepita/pytorch-semseg-master/pred_img'+str(epoch+1)+'.png')
	    if i == 0:
              plt.title("Prediction image")
        #fig1.savefig
        #plt.show()
     
            running_metrics.update(gt, pred)
        plt.show()
        plt.savefig('/home/deepita/pytorch-semseg-master/input_img.png')    
        #fig1.savefig(sys.stdout


        score, class_iou = running_metrics.get_scores()
        for k, v in score.items():
            print(k, v)
        running_metrics.reset()

        if score['Mean IoU : \t'] >= best_iou:
            best_iou = score['Mean IoU : \t']
            state = {'epoch': epoch+1,
                     'model_state': model.state_dict(),
                     'optimizer_state' : optimizer.state_dict(),}
            torch.save(state, "{}_{}_best_model.pkl".format(args.arch, args.dataset))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch', nargs='?', type=str, default='fcn8s', 
                        help='Architecture to use [\'fcn8 etc\']')
    parser.add_argument('--dataset', nargs='?', type=str, default='pascal', 
                        help='Dataset to use [\'BACH etc\']')
    parser.add_argument('--img_rows', nargs='?', type=int, default=256, 
                        help='Height of the input image')
    parser.add_argument('--img_cols', nargs='?', type=int, default=256, 
                        help='Height of the input image')
    parser.add_argument('--n_epoch', nargs='?', type=int, default=100, 
                        help='# of the epochs')
    parser.add_argument('--batch_size', nargs='?', type=int, default=1, 
                        help='Batch Size')
    parser.add_argument('--l_rate', nargs='?', type=float, default=1e-5, 
                        help='Learning Rate')
    parser.add_argument('--feature_scale', nargs='?', type=int, default=1, 
                        help='Divider for # of features to use')
    parser.add_argument('--resume', nargs='?', type=str, default=None,    
                        help='Path to previous saved model to restart from')
    args = parser.parse_args()
    train(args)
