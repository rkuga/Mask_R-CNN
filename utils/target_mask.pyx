import numpy as np
import random
import cv2
from PIL import Image
from StringIO import StringIO
from chainer.links.model.vision.vgg import prepare as vgg_prepare
cimport numpy as np   
DTYPE = np.int32
ctypedef np.int32_t DTYPE_t

def generate_tgt_semantics(np.ndarray[np.float32_t, ndim=5] data_resnet,np.ndarray[np.int32_t, ndim=6] tgt_seg,np.ndarray[np.int32_t, ndim=3] topk_bbox,data,segs,int out_channel):
    cdef int b,k
    cdef np.ndarray[np.int32_t, ndim=1] bbox
    cdef np.ndarray[np.float32_t, ndim=3] image
    cdef np.ndarray[np.float32_t, ndim=2] seg
    cdef float h_orig_ratio,w_orig_ratio
    cdef int x,x2,y,y2
    cdef np.ndarray[np.float32_t, ndim=3] RoI
    cdef np.ndarray[np.float32_t, ndim=2] RoI_float
    cdef np.ndarray[np.int32_t, ndim=2] RoI_t

    for b in range(len(topk_bbox)):
        for k,bbox in enumerate(topk_bbox[b]):
            image = np.asarray(Image.open(StringIO(data[b])).convert('RGB')).astype(np.float32).transpose(2, 0, 1)
            seg = np.asarray(Image.open(StringIO(segs[b]))).astype(np.float32)
            h_orig_ratio=image.shape[1]/224.0
            w_orig_ratio=image.shape[2]/224.0
            x = int(bbox[0]*w_orig_ratio)
            x2 = int(bbox[2]*w_orig_ratio)
            if x > x2:
                x,x2 = x2,x
            y = int(bbox[1]*h_orig_ratio)
            y2 = int(bbox[3]*h_orig_ratio)
            if y > y2:
                y,y2 = y2,y
            if x < 0:
                x=0
            if x2 > image.shape[2]:
                x2=image.shape[2]
            if y<0:
                y=0
            if y2>image.shape[1]:
                y2=image.shape[1]
            RoI = image[:,y:y2,x:x2]
            RoI_float = seg[y:y2,x:x2]
            if RoI.shape[1]==0 or RoI.shape[2]==0:
                data_resnet[k,b,...] = np.zeros((3, 224, 224), dtype=np.float32)
                tgt_seg[k,b,...]=-1
            else:
                data_resnet[k,b,...] = vgg_prepare(RoI)
                RoI_t = cv2.resize(RoI_float,(14,14)).astype(np.int32)
                for i in xrange(out_channel):
                    tgt_seg[k,b,0,i,:,:] = RoI_t[:,:] == i


    return data_resnet,tgt_seg




def generate_tgt_bbox(np.ndarray[np.float32_t, ndim=3] tgt_bbox,np.ndarray[np.int32_t, ndim=2] tgt_cls,np.ndarray[np.int32_t, ndim=3] topk_bbox,gt_bboxs):
    cdef int b,k,max_k,max_b
    cdef np.ndarray[np.int32_t, ndim=1] bbox
    cdef float top_threshold,bottom_threshold,x,x2,y,y2,max_iou,gt_bbox_area,pred_bbox_area,intersection_area
    cdef float mx1,my1,mx2,my2,ex1,ey1,ex2,ey2,tx1,ty1,tx2,ty2
    cdef float intersection_x2,intersection_x1,intersection_y2,intersection_y1,intersection_w,intersection_h,iou
    cdef int index

    for b in range(len(topk_bbox)):
        for k,bbox in enumerate(topk_bbox[b]):
            iou_hist=[]
            index_hist=[]
            top_threshold = 0.7
            bottom_threshold = 0.3
            x,y,x2,y2=bbox

            max_iou=0.0
            max_k=0
            max_b=0
            max_iou_xy = [0,0,0,0]
            for cls_label in gt_bboxs[b]:
                gt_bbox_area = abs((gt_bboxs[b][cls_label][2]-gt_bboxs[b][cls_label][0])*(gt_bboxs[b][cls_label][3]-gt_bboxs[b][cls_label][1]))
                pred_bbox_area = abs((x2-x)*(y2-y))
                intersection_area=0.0
                mx1,my1,mx2,my2 = x,y,x2,y2
                ex1,ey1,ex2,ey2 = gt_bboxs[b][cls_label]
                if mx1 > ex1:
                    tx1,ty1,tx2,ty2 = mx1,my1,mx2,my2
                    mx1,my1,mx2,my2 = ex1,ey1,ex2,ey2
                    ex1,ey1,ex2,ey2 = tx1,ty1,tx2,ty2
                if (mx1 <= ex2 and ex1 <= mx2 and my1 <= ey2 and ey1 <= my2):
                    if mx2 <= ex2:
                        intersection_x2 = mx2
                    else:
                        intersection_x2 = ex2
                    if mx1 <= ex1:
                        intersection_x1 = ex1
                    else:
                        intersection_x1 = mx1
                    if my2 <= ey2:
                        intersection_y2 = my2
                    else:
                        intersection_y2 = ey2
                    if my1 <= ey1:
                        intersection_y1 = ey1
                    else:
                        intersection_y1 = my1
                    intersection_w = abs(intersection_x2-intersection_x1)
                    intersection_h = abs(intersection_y2-intersection_y1)
                    intersection_area = intersection_h*intersection_w
                try:
                    iou = float(intersection_area/(gt_bbox_area+pred_bbox_area-intersection_area))
                except:
                    iou=0.0
                if iou > max_iou:
                    max_iou = iou
                    max_k=0
                    max_b=0
                    max_iou_xy = [gt_bboxs[b][cls_label][0],gt_bboxs[b][cls_label][1],gt_bboxs[b][cls_label][2],gt_bboxs[b][cls_label][3]]
                    if 0 in max_iou_xy:
                        index = max_iou_xy.index(0)
                        max_iou_xy[0]=1


            if max_iou >= top_threshold:
                tgt_bbox[k,b,:]=max_iou_xy
                tgt_cls[k,b]=1
            else:#ToDo:
                tgt_bbox[k,b,:]=max_iou_xy
                # tgt_cls[k,b]=1

        if tgt_cls[:,b].sum()==0:
            tgt_bbox[max_k,b,:]=max_iou_xy
            tgt_cls[max_k,b]=1


    num_bg = np.sum(tgt_cls == 1)
    bg_inds = np.where(tgt_cls == 0)
    if len(bg_inds[0]) > num_bg:
        disable_ind = np.random.choice(len(bg_inds[0]), size=int(len(bg_inds[0]) - num_bg), replace=False)
        disable_inds=([bg_inds[0][disable_ind],bg_inds[1][disable_ind]])
        tgt_cls[disable_inds] = -1

    return tgt_bbox,tgt_cls




