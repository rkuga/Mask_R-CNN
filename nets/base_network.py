from chainer import serializers
from chainer import Variable
import numpy as np
import chainer.functions as F
from chainer import cuda
import os
import cv2
import json
import cPickle as pickle
from PIL import Image


class BaseNetwork(object):

    def __init__(self, epochs, save_every_i, save_every_m):
        self.save_every_i = save_every_i
        self.save_every_m = save_every_m
        self.epochs=epochs
    
    def my_state(self):
        return '%s_'%(self.name)

    
    def save_params(self, epoch):
        print "==> saving state %s" % self.out_model_dir
        serializers.save_hdf5("%s/net_model_classifier_%d.h5"%(self.out_model_dir, epoch),self.network)
    

    def load_state(self,path,epoch):
        print "==> loading state %s epoch %s"%(path,epoch)
        serializers.load_hdf5('./states/%s/net_model_classifier_%s.h5'%(path,epoch), self.network)
        return int(epoch)


    def read_batch(self, perm, batch_index, data_raw):

        data = np.zeros((self.batchsize, self.in_channel, self.input_height, self.input_width), dtype=np.float32)
        label = np.zeros((self.batchsize), dtype=np.int32)

        for j_,j in enumerate(perm[batch_index:batch_index+self.batchsize]):
            data[j_,:,:,:] = data_raw[j][0].astype(np.float32)
            label[j_] = int(data_raw[j][1])

        return data, label
    
    
    def step(self,perm,batch_index, mode, epoch): 
        if mode =='train':
            data, label=self.read_batch(perm,batch_index,self.train_data)
        else:
            data, label=self.read_batch(perm,batch_index,self.test_data)

        data = Variable(cuda.to_gpu(data))
        yl = self.network(data)

        label=Variable(cuda.to_gpu(label))

        L_network = F.softmax_cross_entropy(yl, label)
        A_network = F.accuracy(yl, label)

        if mode=='train':
            self.o_network.zero_grads()
            L_network.backward()
            self.o_network.update()


        return {"prediction": yl.data.get(),
                "current_loss": L_network.data.get(),
                "current_accuracy": A_network.data.get(),
        }

  
    def get_dataset(self, dataset='coco'):
        if dataset=='coco' or dataset=='coco_30k':
            train_ann_path='/data/dataset/MSCOCO/annotations/instances_train2014.json'
            test_ann_path='/data/dataset/MSCOCO/annotations/instances_val2014.json'
            train_image_path='/data/dataset/MSCOCO/train2014/'
            test_image_path='/data/dataset/MSCOCO/val2014/'
            train_seg_path='/data/dataset/MSCOCO/train2014_semantics_categoryID/'
            test_seg_path='/data/dataset/MSCOCO/val2014_semantics_categoryID/'
            self.in_channel=3

            with open('./utils/coco.json', 'r') as fp:
                info = json.load(fp)
            self.palette = np.array(info['palette'], dtype=np.uint8)
            self.out_channel = len(self.palette)
            with open('/data/dataset/MSCOCO/train2014_bboxs/bbox_info.pkl', 'r') as f:
                train_bboxs = pickle.load(f)
            with open('/data/dataset/MSCOCO/val2014_bboxs/bbox_info.pkl', 'r') as f:
                test_bboxs = pickle.load(f)

            f = open(train_ann_path,'r')
            json_data=json.load(f)
            f.close()

            train_label_dataset=[]
            train_data_X={}
            test_data_X={}
            train_data_X=[]
            test_data_X=[]
            train_image_ids=[]            
            
            if self.mode=='train':
                for i,caption_data in enumerate(json_data['annotations']):
                    if dataset=='coco_30k':
                        if i%13!=0:
                            continue
                    # if i%9000!=0:
                    #     continue

                    image_id=caption_data['image_id']
                    bboxs_dict = train_bboxs[image_id]
                    if image_id in train_image_ids:
                        continue
                    img=open(train_image_path+'COCO_train2014_%012d.jpg'%(image_id),'rb').read()
                    try:
                        # if self.stage=='rpn':
                        #     seg_image=0
                        # elif self.stage=='mask':
                        seg_image=open(train_seg_path+'COCO_train2014_%012d.png'%(image_id),'rb').read()
                        # else:
                        #     print 'unrecognized stage'
                    except Exception:
                        continue

                    train_data_X.append((img,seg_image,bboxs_dict))
                    train_image_ids.append(image_id)
                    
            f = open(test_ann_path,'r')
            json_data_test=json.load(f)
            f.close()
            
            count=[]
            test_image_ids=[]
            test_dic={}
            self.vocab_test={}

            for i,caption_data in enumerate(json_data_test['annotations']):
                # break
                # if i==24:
                #     break
                if len(test_data_X)==200:
                    break
                image_id=caption_data['image_id']
                bboxs_dict = test_bboxs[image_id]

                if image_id in test_image_ids:
                    continue
                img=open(test_image_path+'COCO_val2014_%012d.jpg'%(image_id),'rb').read()
                try:
                    # if self.stage=='rpn':
                    #     seg_image=0
                    # else:
                    seg_image=open(test_seg_path+'COCO_val2014_%012d.png'%(image_id),'rb').read()
                except Exception:
                    continue

                test_data_X.append((img,seg_image,bboxs_dict))
                test_image_ids.append(image_id)



        else:
            raise Exception("unrecognized dataset")

        self.out_model_dir ='./states/'+self.my_state()+dataset
        self.out_image_dir = './out_images/'+self.my_state()+dataset

        if self.mode=='test' or self.mode=='generate':
            self.out_image_dir = './test_images/'+self.my_state()+dataset+'/'+self.load[1]+'/'

        if not os.path.exists(self.out_model_dir):
            os.makedirs(self.out_model_dir)
        if not os.path.exists(self.out_image_dir):
            os.makedirs(self.out_image_dir)



        if self.mode=='train':
            print "==> %d training examples" % len(train_data_X)
            print "out_image_dir ==> %s " % self.out_image_dir
            print "out_model_dir ==> %s " % self.out_model_dir
            print "==> %d test examples" % len(test_data_X)
        else:
            print "==> %d test examples" % len(test_data_X)
            print "out_image_dir ==> %s " % self.out_image_dir


        return train_data_X, test_data_X
