import numpy as np
import random
import cv2


cimport numpy as np   
DTYPE = np.int32
ctypedef np.int32_t DTYPE_t

def generate_tgt_bbox(gt_bboxs, np.ndarray[np.double_t, ndim=2] anchors, np.ndarray[np.int32_t, ndim=2] sizes):
    cdef int feature_map_shape = 14
    cdef np.ndarray[np.float32_t, ndim=4] tgt_bbox
    cdef np.ndarray[np.int32_t, ndim=4] tgt_cls
    tgt_bbox = np.zeros((len(gt_bboxs),4*len(anchors),feature_map_shape,feature_map_shape), dtype=np.float32)-1
    tgt_cls = np.zeros((len(gt_bboxs),feature_map_shape,feature_map_shape,len(anchors)), dtype=np.int32)
    cdef int i
    cdef float mx1,my1,mx2,my2
    cdef int num_bg
    cdef float h_ratio,w_ratio,top_threshold,bottom_threshold
    cdef int h,w
    cdef float height,width
    cdef int j_,j
    cdef np.ndarray[np.double_t, ndim=1] anchor
    cdef float x,x2,y,y2,max_iou
    cdef float gt_bbox_area,anchor_area,intersection_area
    cdef float ex1,ey1,ex2,ey2
    cdef float intersection_x2,intersection_x1,intersection_y2,intersection_y1,intersection_w,intersection_h
    cdef float tx1,ty1,tx2,ty2
    cdef float iou
    cdef float h_orig,w_orig,h_orig_ratio,w_orig_ratio
    cdef int index


    for i,gt_bbox in enumerate(gt_bboxs):
        size=sizes[i]
        h_ratio = float(size[0]/14.0)
        w_ratio = float(size[1]/14.0)
        top_threshold = 0.7
        bottom_threshold = 0.3
        iou_hist=[]
        index_hist=[]
        for h in range(7,feature_map_shape+1):
            for w in range(7,feature_map_shape+1):
                height = h*h_ratio
                width = w*w_ratio
                j=0
                for j_,anchor in enumerate(anchors):
                    x = width-anchor[2]/2
                    x2 = width+anchor[2]/2
                    y = height-anchor[3]/2
                    y2 = height+anchor[3]/2
                    
                    max_iou=0.0
                    max_iou_xy = [0,0,0,0]
                    for cls_label in gt_bbox:
                        gt_bbox_area = abs((gt_bbox[cls_label][2]-gt_bbox[cls_label][0])*(gt_bbox[cls_label][3]-gt_bbox[cls_label][1]))
                        anchor_area = abs((x2-x)*(y2-y))
                        mx1,my1,mx2,my2 = x,y,x2,y2
                        ex1,ey1,ex2,ey2 = gt_bbox[cls_label]
                        intersection_area=0.0
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
                        iou = float(intersection_area/(gt_bbox_area+anchor_area-intersection_area))
                        if iou > max_iou:
                            max_iou = iou
                            max_iou_xy = [gt_bbox[cls_label][0],gt_bbox[cls_label][1],gt_bbox[cls_label][2],gt_bbox[cls_label][3]]
                            if 0 in max_iou_xy:
                                index = max_iou_xy.index(0)
                                max_iou_xy[0]=1

                    iou_hist.append(max_iou)
                    index_hist.append((i,h-1,w-1,j_))
                    if max_iou >= 0.7:
                        tgt_cls[i,h-1,w-1,j_] = 1
                        h_orig=size[0]
                        w_orig=size[1]
                        h_orig_ratio=224.0/h_orig
                        w_orig_ratio=224.0/w_orig
                        max_iou_xy = [gt_bbox[cls_label][0]*w_orig_ratio,gt_bbox[cls_label][1]*h_orig_ratio,gt_bbox[cls_label][2]*w_orig_ratio,gt_bbox[cls_label][3]*h_orig_ratio]
                        tgt_bbox[i,j:j+4,h-1,w-1] = np.array(max_iou_xy)
                    else:
                        pass
                        # tgt_cls[i,h-1,w-1,j_] = 0
                    j+=4

        if len(iou_hist)>0:
            if max(iou_hist)<0.7:
                index = iou_hist.index(max(iou_hist))
                i,h,w,j = index_hist[index]
                tgt_cls[i,h,w,j] = 1
                tgt_bbox[i,j:j+4,h,w] = np.array(max_iou_xy)

    # num_fg = int(0.5 * len(gt_bboxs))
    # fg_inds = np.where(tgt_cls == 1)
    # if len(fg_inds[0]) > num_fg:
    #     disable_ind = np.random.choice(len(fg_inds[0]), size=int(len(fg_inds[0]) - num_fg), replace=False)
    #     disable_inds=([fg_inds[0][disable_ind],fg_inds[1][disable_ind],fg_inds[2][disable_ind],fg_inds[3][disable_ind]])
    #     tgt_cls[disable_inds] = -1

    # num_bg = len(gt_bboxs) - np.sum(tgt_cls == 1)
    num_bg = np.sum(tgt_cls == 1)
    bg_inds = np.where(tgt_cls == 0)
    if len(bg_inds[0]) > num_bg:
        disable_ind = np.random.choice(len(bg_inds[0]), size=int(len(bg_inds[0]) - num_bg), replace=False)
        disable_inds=([bg_inds[0][disable_ind],bg_inds[1][disable_ind],bg_inds[2][disable_ind],bg_inds[3][disable_ind]])
        tgt_cls[disable_inds] = -1

    inside_ind=np.where(tgt_bbox!=0)
    return tgt_bbox,tgt_cls,inside_ind




