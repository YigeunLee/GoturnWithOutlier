import numpy as np
import goturn
import train
import gaussian_classifier
import tensorflow as tf
import os
import glob
import random
import skimage
import scipy
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from sklearn import svm
import cv2
#crop 시 width,height x2를 할 것#
#찾은위치에서 또 찾을 때 width,height x2곱해서 ..

batch_size = 4
learning_rate = 0.00001
n_epochs = 1500
bbox_tensor_size = 4 # x,y,width,height
traing_path = r'E:\개인 프로젝트\텐서플로\goturn\test_set'
val_path = r'E:\개인 프로젝트\텐서플로\goturn\test_set\validation'
def load_model(path):
    return 0

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

def get_validate(train_set_filepath):
    prev_images = []
    curr_images = []
    result_images = []

    files = os.listdir(path=val_path)
        
    prev_path = os.path.join(val_path,'prev')
    prev_files = os.listdir(prev_path)
    prev_files = sorted(prev_files,key=lambda x:int(x.split('_')[0]))

    curr_path = os.path.join(val_path,'curr')
    curr_files = os.listdir(curr_path)
    curr_files = sorted(curr_files,key=lambda x:int(x.split('_')[0]))

    result_path = os.path.join(val_path,'result')
    result_files = os.listdir(result_path)
    result_files = sorted(result_files,key=lambda x:int(x.split('_')[0]))

    for p in prev_files:
        prev_image = scipy.misc.imresize(scipy.misc.imread(os.path.join(prev_path,p)), (224,224))
        prev_images.append(prev_image)
    for c in curr_files:
        curr_image = scipy.misc.imresize(scipy.misc.imread(os.path.join(curr_path,c)), (224,224))
        curr_images.append(curr_image)
    for r in result_files:
        result_image = scipy.misc.imread(os.path.join(result_path,r))
        result_images.append(result_image)

    return np.array(prev_images,dtype=np.float32) / 255.0 ,np.array(curr_images,dtype=np.float32) / 255.0 , np.array(result_images)

def get_batch_func(train_set_filepath):
    prev_images = []
    curr_images = []
    bbox_label = []

    files = os.listdir(path=train_set_filepath)
        
    prev_path = os.path.join(train_set_filepath,'prev')
    prev_files = os.listdir(prev_path)
    prev_files = sorted(prev_files,key=lambda x:int(x.split('_')[0]))

    curr_path = os.path.join(train_set_filepath,'curr')
    curr_files = os.listdir(curr_path)
    curr_files = sorted(curr_files,key=lambda x:int(x.split('_')[0]))
    
    lb_path = os.path.join(train_set_filepath,'label')
    lb_files = os.listdir(lb_path)
    
    rnd_idx = random.randrange(0,len(curr_files) - batch_size)

    for p in prev_files[rnd_idx:(rnd_idx + batch_size)]:
        prev_image = scipy.misc.imresize(scipy.misc.imread(os.path.join(prev_path,p)), (224,224))
        prev_images.append(prev_image)
    for c in curr_files[rnd_idx:(rnd_idx + batch_size)]:
        curr_image = scipy.misc.imresize(scipy.misc.imread(os.path.join(curr_path,c)), (224,224))
        curr_images.append(curr_image)
        
    f = open(os.path.join(lb_path,lb_files[0]))
    bbox_list = f.read()
    bbox_list = bbox_list.split('\n')
   
    for l in bbox_list[rnd_idx:(rnd_idx + batch_size)]:
        bbox = l.split(',')[1:5]
        bbox_label.append([int(bbox[0]) /1720.0,int(bbox[1]) / 880.0,int(bbox[2]) / 1920.0,int(bbox[3]) / 1080.0])

    return np.array(prev_images,dtype=np.float32) / 255.0 ,np.array(curr_images,dtype=np.float32) / 255.0 , np.array(bbox_label)
    
def play():
    cap = cv2.VideoCapture(r'E:\회사소스\스마트모니터링 자료\연구실\영상\1_JA_1B-06_[BEIGE]_01_[2421].avi')
    cnt = 1
    while(cap.isOpened()):
        ret,frame = cap.read()
        if ret==True:
            #frame = cv2.flip(frame,0)

            # write the flipped frame

            '''
            cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            '''
            yield cnt,frame
            cv2.waitKey(10)
            cnt += 1
            
        else:
            break
    

def write_train_box(train_box_data):
    train_set_filepath = r'E:\개인 프로젝트\텐서플로\goturn\test_set'
    files = os.listdir(path=train_set_filepath)
    curr_path = os.path.join(train_set_filepath,'label')
    curr_files = os.listdir(curr_path)
    fa = open(os.path.join(curr_path,'surging_gt(1).txt'),'w')

    for i in range(0, len(train_box_data) - 1):
        fa.write('0,{0},{1},{2},{3}\n'.format(train_box_data[i][0],train_box_data[i][1],train_box_data[i][2],train_box_data[i][3]))

    fa.close()

k = 2
def get_resized_roiRectangle(prev_box):
    resized_y = 0
    resized_height = prev_box[3]
    resized_x = 0
    resized_width = prev_box[2]
    
    if (prev_box[1] + prev_box[3]) >= 1080.0:
        if prev_box[3] >= 1080.0:
            resized_y = 0
            resized_height = 1080.0 - 1
        else:
            resized_y = (prev_box[1] - np.abs((prev_box[1] + resized_height) - 1080.0)) - 1
            
            if resized_y< 0:
                resized_y = 0
    else:
        resized_y = prev_box[1]

    if (prev_box[0] + prev_box[2]) >= 1920.0:
        if prev_box[2] >= 1920.0:
            resized_x = 0
            resized_width = 1920.0 - 1
        else:
            resized_x = (prev_box[0] - np.abs((prev_box[0] + resized_width) - 1920.0)) - 1
            
            if resized_x< 0:
                resized_x = 0
    else:
        resized_x = prev_box[0]
            
    return [[int(resized_x),int(resized_y),int(resized_width),int(resized_height)]]
    

def get_position_diff(prev_box,result_box):
    return [[np.abs(prev_box[0,0] - result_box[0,0]),np.abs(prev_box[0,1] - result_box[0,1])]]

def predict(cnt,clf,pred_target,pos_ranges,result_box):
    val = clf.predict(pred_target)

    if cnt >= len(pos_ranges):
        return 1


    if pos_ranges[cnt][0][0] < result_box[0,0] and pos_ranges[cnt][0][1] > result_box[0,0] and pos_ranges[cnt][1][0] < result_box[0,1] and pos_ranges[cnt][1][1] > result_box[0,1]:
        return 1 * val
    elif val == -1:
        return -2
    else:
        return -1
    return val

if __name__ == "__main__":
    reset_graph()
    
    prev_frame = tf.placeholder(tf.float32, shape=[None, 224, 224,3] ,name="prev_frame")
    curr_frame = tf.placeholder(tf.float32, shape=[None, 224, 224,3] ,name="curr_frame")
    label_bbox = tf.placeholder(tf.float32, shape=[None,bbox_tensor_size], name="bbox") # batch,[x,y,width,height]
    training = tf.placeholder_with_default(False, shape=[], name='training')
    clf,pos_ranges = gaussian_classifier.get_classifier()
    train_box_data = []
    cnt = 0
    with tf.Session() as sess:
        layers = goturn.layers(prev_frame,curr_frame)
        pred_layers = goturn.prediction(layers,training)
        result_box = np.array([[0,0,0,0]])
        training_op,loss,acc = train.optimize(pred_layers,label_bbox,learning_rate)
        #train.train(sess,prev_frame=prev_frame,curr_frame=curr_frame,label_bbox=label_bbox,training=training,
        #            training_op=training_op,loss=loss,acc=acc,n_epochs=n_epochs,get_batch_func=get_batch_func,batch_size=batch_size,train_set_filepath=traing_path,training_val=True)
        saver = tf.train.Saver()
        saver.restore(sess,r'E:\개인 프로젝트\텐서플로\goturn.ckpt')
        #saver.save(sess,r'E:\개인 프로젝트\텐서플로\goturn.ckpt')
        
        #prev_val_frame,curr_val_frame,result_val_image # = get_validate(traing_path)
        prev_box = np.array([[560,530,700,450]])
        prev = None
        color = [0,255,0]
        for pos,frame in play():#zip(prev_val_frame,curr_val_frame,result_val_image):

            if pos > 411:
                break
            
            if pos % 5 == 1:
              
                roi_rect = get_resized_roiRectangle(np.array([prev_box[0,0],prev_box[0,1],prev_box[0,2] * k,prev_box[0,3] * k]))
              
                if prev is None:
                    prev = frame[roi_rect[0][1]:(roi_rect[0][1] + roi_rect[0][3]),roi_rect[0][0]:(roi_rect[0][0] + roi_rect[0][2])]
                curr = frame[roi_rect[0][1]:(roi_rect[0][1] + roi_rect[0][3]),roi_rect[0][0]:(roi_rect[0][0] + roi_rect[0][2])]
              
                curr = cv2.resize(curr, dsize=(224, 224))
                prev = cv2.resize(prev, dsize=(224, 224))

                curr = curr / 255.0
                prev = prev / 255.0

                pred_bbox = sess.run(pred_layers,feed_dict={prev_frame: [curr], curr_frame: [curr],training:False})
                #print('regression result => ' + str(pred_bbox[0,0] * 1720.0) + ',' + str(pred_bbox[0,1] * 880.0) + ',' + str(pred_bbox[0,2] * 1920.0) + ',' + str(pred_bbox[0,3] * 1080.0))
                result_box = np.array([[int(pred_bbox[0,0] * 1720.0),int(pred_bbox[0,1] * 880.0),int(pred_bbox[0,2] * 1920.0),int(pred_bbox[0,3] * 1080.0)]])
                train_box_data.append([int(pred_bbox[0,0] * 1720.0),int(pred_bbox[0,1] * 880.0),int(pred_bbox[0,2] * 1920.0),int(pred_bbox[0,3] * 1080.0)])

                pred_data = get_position_diff(prev_box,result_box)
                pred = predict(cnt,clf,pred_data,pos_ranges,result_box)
                if pred == -1:
                    color = [0,0,255]
                    print(pred_data)
                elif pred == -2:
                    color = [0,255,255]
                    print('range=>' + str(pos_ranges[cnt]) + ',result=>' + str(result_box))
                else:
                    color = [0,255,0]
                    
                cnt += 1
         
            cv2.rectangle(frame,(result_box[0,0],result_box[0,1]),(result_box[0,0] + result_box[0,2],result_box[0,1] + result_box[0,3]),color,3)
            disp = cv2.resize(frame, dsize=(640, 480))
            cv2.imshow('tracker',disp)
            
            if pos % 5 == 1:
                prev_box = result_box
                prev = curr
            
    write_train_box(train_box_data)
