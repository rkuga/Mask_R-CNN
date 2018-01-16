import numpy as np
import random
import cv2
import chainer.functions as F
cimport numpy as np   
DTYPE = np.int32
ctypedef np.int32_t DTYPE_t


def rpn_topk_loop(int batchsize, int topk, np.ndarray[np.int32_t, ndim=3] topk_bbox, np.ndarray[np.float32_t, ndim=3] pred_bbox,np.ndarray[np.long_t, ndim=2] order):
    cdef int b,t
    for b in range(batchsize):
        for t in range(topk):
            topk_bbox[b,t,:] = pred_bbox[b,order[b][t],:]
            
    return topk_bbox

def cnn_topk_loop(int topk, cnn, data_resnet, train):
    cdef int k 
    for k in range(topk):
        if k==0:
            topk_cls_label,topk_box,topk_mask=cnn(data_resnet[k], train=train, test=not train)
            topk_cls_label=F.expand_dims(topk_cls_label,axis=0)
            topk_box=F.expand_dims(topk_box,axis=0)
            topk_mask=F.expand_dims(topk_mask,axis=0)
        else:
            cls_label,box,mask=cnn(data_resnet[k], train=train, test=not train)
            cls_label=F.expand_dims(cls_label,axis=0)
            box=F.expand_dims(box,axis=0)
            mask=F.expand_dims(mask,axis=0)
            topk_cls_label = F.vstack((topk_cls_label,cls_label))
            topk_box = F.vstack((topk_box,box))
            topk_mask = F.vstack((topk_mask,mask))

    return topk_cls_label,topk_box,topk_mask