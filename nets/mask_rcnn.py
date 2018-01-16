import numpy as np
import cv2
import os
import chainer
from chainer import cuda
from chainer import optimizers
from chainer import serializers
from chainer import Variable
import chainer.functions as F
import chainer.links as L
from base_network import BaseNetwork
import random
import time
from PIL import Image
import pyximport; pyximport.install()
from StringIO import StringIO
from progressbar import ProgressBar
import utils.My_mean_absolute_error as My_Mean_absolute_error
import utils.My_SoftmaxCrossEntropy as My_Softmax_cross_entropy
from utils.anchor import generate_anchors as generate_anchors
from utils.target_rpn import generate_tgt_bbox as generate_tgt_bbox
from utils.topk_loop import rpn_topk_loop as rpn_topk_loop
from utils.topk_loop import cnn_topk_loop as cnn_topk_loop
from utils.target_mask import generate_tgt_semantics as generate_tgt_semantics
from utils.target_mask import generate_tgt_bbox as generate_tgt_bbox_mask
from chainer.links.model.vision.vgg import prepare as vgg_prepare
from chainer.links.model.vision.resnet import prepare as resnet_prepare

class VGG16Layer(chainer.Chain):
    def __init__(self):
        w = chainer.initializers.HeNormal()
        super(VGG16Layer, self).__init__(
            vgg = L.VGG16Layers(),
            )

    def __call__(self, x):
        h = self.vgg(x, layers=['conv5_3'])
        return h['conv5_3']

class RPN(chainer.Chain):
    def __init__(self, in_channel):
        self.n_anchors=9
        super(RPN, self).__init__(
            conv1=L.Convolution2D(in_channel, 512, 3, stride=1, pad=1),
            conv1_bn = L.BatchNormalization(512),
            conv2=L.Convolution2D(512, 2*self.n_anchors, 1, stride=1, pad=0),
            conv3=L.Convolution2D(512, 4*self.n_anchors, 1, stride=1, pad=0),

        )
        
    def __call__(self, z, train=True, test=False):
        h = F.relu(self.conv1_bn(self.conv1(z), test=test))
        h = F.dropout(h, train=train)
        cls_score = self.conv2(h).reshape(z.data.shape[0],2,self.n_anchors,z.data.shape[2],z.data.shape[3])
        bbox = self.conv3(h)

        return cls_score,bbox

class CNN(chainer.Chain):
    def __init__(self,out_channel):
        self.out_channel=out_channel
        super(CNN, self).__init__(
            resnet=L.VGG16Layers(),#ToDo:use resnet
            fc_class = L.Linear(512,2),
            fc_box = L.Linear(512,4),
            mask_1 = L.Deconvolution2D(512, 256, 2, stride=2, pad=0),
            conv1_bn = L.BatchNormalization(256),
            mask_2 = L.Convolution2D(256, out_channel, 3, stride=1, pad=1),


        )
        
    def __call__(self, x, train=True, test=False):
        h = self.resnet(x, layers=['pool5'])
        h = h['pool5']
        ave = F.sum(h,axis=(2,3))/49
        cls_label = self.fc_class(ave)
        box = self.fc_box(ave)
        h = F.dropout(h, train=train)
        h = F.relu(self.conv1_bn(self.mask_1(h), test=test))
        mask = self.mask_2(h).reshape(x.data.shape[0],1,self.out_channel,14,14)

        return cls_label,box,mask




class Network(BaseNetwork):
    def __init__(self,gpu,batchsize,dataset,net,mode,epochs,save_every_i,save_every_m,size,load,stage, **kwargs):
        super(Network, self).__init__(epochs,save_every_i,save_every_m)
        print "==> not used params in this network:", kwargs.keys()
        print "building ..."
        self.input_height=size
        self.input_width=size
        self.net = net
        self.mode=mode
        self.load=load
        self.dataset=dataset
        self.stage=stage
        self.train_data, self.test_data=self.get_dataset(dataset)
        self.anchors = generate_anchors()
        self.n_anchors = len(self.anchors)

        self.vgg = VGG16Layer()
        self.cnn = CNN(self.out_channel)
        self.rpn = RPN(512)

        self.xp = cuda.cupy
        cuda.get_device(gpu).use()

        self.vgg.to_gpu()
        self.cnn.to_gpu()
        self.rpn.to_gpu()

        self.o_cnn = optimizers.MomentumSGD(lr=0.02)
        self.o_cnn.setup(self.cnn)
        self.o_cnn.add_hook(chainer.optimizer.WeightDecay(0.0001))

        self.o_rpn = optimizers.MomentumSGD(lr=0.02)
        self.o_rpn.setup(self.rpn)
        self.o_rpn.add_hook(chainer.optimizer.WeightDecay(0.0001))

        self.batchsize=batchsize


    def my_state(self):
        return '%s_'%(self.net)

    def read_batch(self, perm, batch_index,data_raw):
        data_vgg = np.zeros((self.batchsize, self.in_channel, self.input_height, self.input_width), dtype=np.float32)
        # data = np.zeros((self.batchsize, self.in_channel, self.input_height, self.input_width), dtype=np.float32)
        data=[]
        segs=[]
        t = np.zeros((self.batchsize, self.input_height, self.input_width), dtype=np.int32)
        size = np.zeros((self.batchsize, 2), dtype=np.int32)
        bboxs = []
        for j_,j in enumerate(perm[batch_index:batch_index+self.batchsize]):
            image = np.asarray(Image.open(StringIO(data_raw[j][0])).convert('RGB')).astype(np.float32).transpose(2, 0, 1)
            size[j_] = np.array(image.shape[1:])
            image_vgg = vgg_prepare(image)
            # image_resnet = vgg_prepare(image)
            data_vgg[j_,:,:,:] = image_vgg
            data.append(data_raw[j][0])
            segs.append(data_raw[j][1])

            if self.stage!='rpn':
                label = np.asarray(Image.open(StringIO(data_raw[j][1]))).astype(np.float32)
                label = cv2.resize(label, (self.input_height,self.input_width))  
                t[j_,:,:] = label[:,:]   
            bboxs.append(data_raw[j][2])         

        
        return data_vgg, data, segs, t, size, bboxs

    def step(self,perm,batch_index,mode,epoch): 
        if mode=='train':
            data_vgg, data, segs, t, size, gt_bboxs=self.read_batch(perm,batch_index,self.train_data)
            train=True
        else :
            data_vgg, data, segs, t, size, gt_bboxs=self.read_batch(perm,batch_index,self.test_data)
            train=False

        data_vgg = Variable(cuda.to_gpu(data_vgg))
        t=Variable(cuda.to_gpu(t))
        h = self.vgg(data_vgg)
        h = h.data
        h = Variable(h)


        if self.stage=='rpn':
            pred_cls_score,pred_bbox = self.rpn(h, train=train, test=not train)
            tgt_bbox,tgt_cls,inside_ind = generate_tgt_bbox(gt_bboxs, self.anchors, size)
            tgt_bbox=Variable(cuda.to_gpu(tgt_bbox))
            tgt_cls=Variable(cuda.to_gpu(tgt_cls.transpose(0,3,1,2)))

            L_rpn_bbox = My_Mean_absolute_error.mean_absolute_error(pred_bbox, tgt_bbox)
            L_rpn_cls = F.softmax_cross_entropy(pred_cls_score, tgt_cls,ignore_label=-1)
            A_rpn = F.accuracy(pred_cls_score, tgt_cls,ignore_label=-1)
            if mode=='train':
                self.rpn.cleargrads()
                L_rpn_bbox.backward()
                L_rpn_cls.backward()
                self.o_rpn.update()


            return {"prediction": pred_cls_score.data.get(),
                    "current_loss": L_rpn_bbox.data.get(),
                    "current_accuracy": A_rpn.data.get(),
            }
        elif self.stage=='mask':
            pred_cls_score,pred_bbox = self.rpn(h, train=False, test=True)

            topk=5
            topk_bbox = self.extract_topk_region(pred_cls_score, pred_bbox, topk)
            data_resnet = np.zeros((topk,self.batchsize, 3, self.input_height, self.input_width), dtype=np.float32)
            tgt_seg = np.zeros((topk,self.batchsize, 1, self.out_channel, 14, 14), dtype=np.int32)
            tgt_cls = np.zeros((topk,self.batchsize), dtype=np.int32)
            tgt_bbox = np.zeros((topk,self.batchsize, 4), dtype=np.float32)-1
            data_resnet,tgt_seg = generate_tgt_semantics(data_resnet,tgt_seg,topk_bbox,data,segs,self.out_channel)

            data_resnet = Variable(cuda.to_gpu(data_resnet))
            tgt_seg = Variable(cuda.to_gpu(tgt_seg))
            for k in range(topk):
                if k==0:
                    topk_cls_label,topk_box,topk_mask=self.cnn(data_resnet[k], train=train, test=not train)
                    topk_cls_label=F.expand_dims(topk_cls_label,axis=0)
                    topk_box=F.expand_dims(topk_box,axis=0)
                    topk_mask=F.expand_dims(topk_mask,axis=0)
                else:
                    cls_label,box,mask=self.cnn(data_resnet[k], train=train, test=not train)
                    cls_label=F.expand_dims(cls_label,axis=0)
                    box=F.expand_dims(box,axis=0)
                    mask=F.expand_dims(mask,axis=0)
                    topk_cls_label = F.vstack((topk_cls_label,cls_label))
                    topk_box = F.vstack((topk_box,box))
                    topk_mask = F.vstack((topk_mask,mask))
            # topk_cls_label,topk_box,topk_mask = cnn_topk_loop(topk,self.cnn,data_resnet,train) 

            tgt_bbox,tgt_cls = generate_tgt_bbox_mask(tgt_bbox,tgt_cls,topk_bbox,gt_bboxs)

            tgt_bbox = Variable(cuda.to_gpu(tgt_bbox))
            tgt_cls = Variable(cuda.to_gpu(tgt_cls))
            L_cnn_bbox = My_Mean_absolute_error.mean_absolute_error(topk_box, tgt_bbox)

            topk_cls_label=F.transpose(topk_cls_label,axes=(0,2,1))
            L_cnn_cls = My_Softmax_cross_entropy.softmax_cross_entropy(topk_cls_label, tgt_cls,ignore_label=-1)
            A_cnn = F.accuracy(topk_cls_label, tgt_cls,ignore_label=-1)
            L_cnn_mask = F.sigmoid_cross_entropy(topk_mask,tgt_seg)

            if mode=='train':
                self.cnn.cleargrads()
                L_cnn_bbox.backward()
                L_cnn_cls.backward()
                L_cnn_mask.backward()
                self.o_cnn.update()


            return {"prediction": topk_cls_label.data.get(),
                    "current_loss": L_cnn_mask.data.get(),
                    "current_accuracy": A_cnn.data.get(),
            }



    def generate(self,epoch):
        batchsize=1
        topk=5
        if batchsize>len(self.test_data):
            batchsize=len(self.test_data)
        index = random.randint(0,len(self.test_data)-1)
        data_vgg = np.zeros((batchsize, self.in_channel, self.input_height, self.input_width), dtype=np.float32)
        pred_bbox = np.zeros((batchsize,4*len(self.anchors),14,14), dtype=np.float32)
        pred_cls_score = np.zeros((batchsize,2,14,14,len(self.anchors)), dtype=np.float32)
        topk_bbox = np.zeros((batchsize,topk,4), dtype=np.int32)
        data=[]
        segs=[]

        image_origin = np.asarray(Image.open(StringIO(self.test_data[index][0])).convert('RGB')).astype(np.float32).transpose(2, 0, 1)
        image = vgg_prepare(image_origin)
        data_vgg[0,:,:,:] = image
        data.append(self.test_data[index][0])
        segs.append(self.test_data[index][1])
        
        data_vgg = Variable(cuda.to_gpu(data_vgg))
        h = self.vgg(data_vgg)
        pred_cls_score,pred_bbox = self.rpn(h, train=False, test=True)
        pred_cls_score_o=pred_cls_score.data.copy()
        pred_bbox_o=pred_bbox.data.copy()

        pred_cls_score = pred_cls_score[:,1,:,:,:]
        pred_cls_score = pred_cls_score.reshape((batchsize,-1)).data.get()

        order = pred_cls_score.argsort(axis=1)[::-1][:,:topk]

        pred_bbox = pred_bbox.reshape((batchsize,4,14,14,len(self.anchors)))
        pred_bbox = pred_bbox.reshape((batchsize,4,-1))
        pred_bbox = pred_bbox.transpose(0,2,1)

        for b in range(order.shape[0]):
            for t in range(topk):
                topk_bbox[b,t,:] = pred_bbox[b,order[b][t],:].data.get()

        # topk_bbox = pred_bbox[order].data.get()

        image_origin = image_origin.transpose(1,2,0)[:,:,::-1].astype(np.int32)
        h_orig=image_origin.shape[0]
        w_orig=image_origin.shape[1]
        h_orig_ratio=h_orig/224.0
        w_orig_ratio=w_orig/224.0
        if self.stage=='rpn':
            for bbox in topk_bbox[0]:
                x,y,x2,y2=int(bbox[0]*w_orig_ratio),int(bbox[1]*h_orig_ratio),int(bbox[2]*w_orig_ratio),int(bbox[3]*h_orig_ratio)
                if x>x2:
                    x,x2=x2,x
                if y>y2:
                    y,y2=y2,y
                cv2.rectangle(image_origin, (x,y), (x2,y2), (255,0,0), 3)
            cv2.imwrite('%s/%d_bbox.png'%(self.out_image_dir, epoch), image_origin)
        elif self.stage=='mask':
            data_resnet = np.zeros((topk,batchsize, 3, self.input_height, self.input_width), dtype=np.float32)
            tgt_seg = np.zeros((topk,batchsize, 1, self.out_channel, 14, 14), dtype=np.int32)

            data_resnet,tgt_seg = generate_tgt_semantics(data_resnet,tgt_seg,topk_bbox,data,segs,self.out_channel)

            data_resnet = Variable(cuda.to_gpu(data_resnet))

            for k in range(topk):
                if k==0:
                    topk_cls_label,topk_box,topk_mask=self.cnn(data_resnet[k], train=False, test=True)
                    topk_mask=F.expand_dims(topk_mask,axis=0)
                else:
                    cls_label,box,mask=self.cnn(data_resnet[k], train=False, test=True)
                    mask=F.expand_dims(mask,axis=0)
                    topk_mask = F.vstack((topk_mask,mask))#(topk,self.batchsize, 1, self.out_channel, 14, 14)

            pred_seg = np.zeros((image_origin.shape[0],image_origin.shape[1]), dtype=np.int32)
            topk_mask=topk_mask.data.get()
            for k in range(topk):
                seg_image=np.argmax(topk_mask[k,0,0,:,:,:],axis=0).astype(np.float32)
                x,y,x2,y2=int(topk_bbox[0][k][0]*w_orig_ratio),int(topk_bbox[0][k][1]*h_orig_ratio),int(topk_bbox[0][k][2]*w_orig_ratio),int(topk_bbox[0][k][3]*h_orig_ratio)
                if x>x2:
                    x,x2=x2,x
                if y>y2:
                    y,y2=y2,y
                if y > y2:
                    y,y2 = y2,y
                if x < 0:
                    x=0
                if x2 > image_origin.shape[1]:
                    x2=image_origin.shape[1]
                if y<0:
                    y=0
                if y2>image_origin.shape[0]:
                    y2=image_origin.shape[0]
                # if abs(x2-x)==0 or abs(y2-y)==0:
                #     continue
                seg_image=cv2.resize(seg_image,(abs(x2-x),abs(y2-y))).astype(np.int32)
                try:
                    pred_seg[y:y2,x:x2]=seg_image
                except ValueError:
                    continue
            color_image = self.palette[pred_seg]
            cv2.imwrite('%s/%d_mask.png'%(self.out_image_dir, epoch), color_image)

    def test(self):
        batchsize=1
        topk = 12
        p = ProgressBar()
        for i_  in p(range(0,len(self.test_data),batchsize)): 
            data = np.zeros((batchsize, self.in_channel, self.input_height, self.input_width), dtype=np.float32)
            pred_bbox = np.zeros((batchsize,4*len(self.anchors),14,14), dtype=np.float32)
            pred_cls_score = np.zeros((batchsize,2,14,14,len(self.anchors)), dtype=np.float32)
            topk_bbox = np.zeros((batchsize,topk,4), dtype=np.int32)            
            for j in xrange(batchsize):
                image_origin = np.asarray(Image.open(StringIO(self.test_data[i_+j][0])).convert('RGB')).astype(np.float32).transpose(2, 0, 1)
                image = vgg_prepare(image_origin)
                data[j,:,:,:] = image

            data = Variable(cuda.to_gpu(data))
            h = self.vgg(data)
            pred_cls_score,pred_bbox = self.rpn(h, train=False, test=True)

            pred_cls_score = pred_cls_score[:,1,:,:,:]
            pred_cls_score = pred_cls_score.reshape((batchsize,-1)).data.get()

            order = pred_cls_score.argsort(axis=1)[::-1][:,:topk]

            pred_bbox = pred_bbox.reshape((batchsize,4,14,14,len(self.anchors)))
            pred_bbox = pred_bbox.reshape((batchsize,4,-1))
            pred_bbox = pred_bbox.transpose(0,2,1)

            for b in range(order.shape[0]):
                for t in range(topk):
                    topk_bbox[b,t,:] = pred_bbox[b,order[b][t],:].data.get()

            # topk_bbox = pred_bbox[order].data.get()

            image_origin = image_origin.transpose(1,2,0)[:,:,::-1].astype(np.int32)
            h_orig=image_origin.shape[0]
            w_orig=image_origin.shape[1]
            h_orig_ratio=h_orig/224.0
            w_orig_ratio=w_orig/224.0
            for bbox in topk_bbox[0]:
                cv2.rectangle(image_origin, (int(bbox[0]*w_orig_ratio),int(bbox[1]*h_orig_ratio)), (int(bbox[2]*w_orig_ratio),int(bbox[3]*h_orig_ratio)), (255,0,0), 3)
            cv2.imwrite('%s/%d_vis.png'%(self.out_image_dir, i_), image_origin)

    def extract_topk_region(self,pred_cls_score,pred_bbox,topk):
        topk_bbox = np.zeros((self.batchsize,topk,4), dtype=np.int32)
        pred_cls_score = pred_cls_score[:,1,:,:,:]
        pred_cls_score = pred_cls_score.reshape((self.batchsize,-1)).data.get()

        order = pred_cls_score.argsort(axis=1)[::-1][:,:topk]

        pred_bbox = pred_bbox.reshape((self.batchsize,4,14,14,len(self.anchors)))
        pred_bbox = pred_bbox.reshape((self.batchsize,4,-1))
        pred_bbox = pred_bbox.transpose(0,2,1).data.get()

        topk_bbox=rpn_topk_loop(order.shape[0],topk,topk_bbox,pred_bbox,order)

        return topk_bbox

    def save_params(self, epoch):
        print "==> saving state %s" % self.out_model_dir
        serializers.save_hdf5("%s/rpn_model_%d.h5"%(self.out_model_dir, epoch),self.rpn)
        serializers.save_hdf5("%s/cnn_model_%d.h5"%(self.out_model_dir, epoch),self.cnn)


    def load_state(self,path,epoch):
        print "==> loading state %s epoch %s"%(path,epoch)
        serializers.load_hdf5('%s/rpn_model_%s.h5'%(self.out_model_dir,epoch), self.rpn)
        serializers.load_hdf5('%s/cnn_model_%s.h5'%(self.out_model_dir,epoch), self.cnn)
        return int(epoch)
