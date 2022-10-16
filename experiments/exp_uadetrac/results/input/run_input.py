import os
import numpy as np
import sys
from tqdm import tqdm
sys.path.append('/raid/workspace/zhangyinan/s3backup/alphaPrime/uper/')
from models.input_model.reducto import img_diff

frames_dir = '/raid/workspace/zhangyinan/Datasets/UA_Detrac/Insight-MVT_Annotation_Test/'

img_list_path = '/raid/workspace/zhangyinan/Datasets/UA_Detrac/processdata/results/sampled_frames_1.txt'

interval = 1
winlen = 30
weak_pred_path = '../../preds_results/labels_preds_mobilenet_'+str(interval)+'.npy'
strong_pred_path = '../../preds_results/labels_preds_resnet_'+str(interval)+'.npy'

save_upres_path = './results/input/input_winlen'+str(winlen)+'_upload_results.npy'

def cal_acc(small_pred,large_pred,upframes):
    right_num = 0
    for i in range(len(small_pred)):
        if i in upframes:
            right_num+=1
        else:
            if small_pred[i]==large_pred[i]:
                right_num+=1
    acc = right_num/len(small_pred)
    return acc

def predict_with_input(datanums,fnames,small_pred,large_pred,target_acc):
    '''根据输入，文件列表，进行预测'''
    fpaths = [frames_dir+name for name in fnames]
    input_diff = [0.]
    input_diff.extend(img_diff(fpaths,'area'))

    upload=[]
    diff_sort = [(input_diff[i],i) for i in range(datanums)]
    diff_sort.sort(key=lambda x:x[0])
    for i in range(datanums):
        upframes = [x[1] for x in diff_sort[0:i]]
        acc = cal_acc(small_pred,large_pred,upframes)
        if acc >= target_acc:
            upload = [1 if j in upframes else 0 for j in range(datanums)]
            break
    # sys acc
    return acc,upload

def upload_with_input(target_acc=0.9):
    '''基于input预测上传结果，一个小段是25是因为帧率是25fps'''
    with open(img_list_path,'r') as f:
        imglist = f.read().splitlines()
    weak_preds_label = np.load(weak_pred_path)
    strong_preds_label = np.load(strong_pred_path)

    imgnums = len(imglist)
    prednums = len(weak_preds_label)
    print(imgnums,prednums)
    if imgnums!=prednums:
        print('model preds num unmatch actual imgs!')
        return None
    uploads_res = []
    for i in tqdm(range(0,imgnums,winlen)):
        rightind = min(i+winlen,imgnums)
        fnames = imglist[i:rightind]
        small_pred = weak_preds_label[i:rightind]
        large_pred = strong_preds_label[i:rightind]
        datanums = rightind-i

        winacc,winupload = predict_with_input(datanums,fnames,small_pred,large_pred,target_acc)
        uploads_res.extend(winupload)
    uploads_res = np.array(uploads_res,dtype=np.int32)
    np.save(save_upres_path,uploads_res)
    
def get_nobalance_input():
    small_pred = np.load(weak_pred_path)
    large_pred = np.load(strong_pred_path)

    test_input = np.load('./input_winlen25_upload_results.npy')

    tlens = len(test_input)

    start = 30000
    lengths  = 20000
    end = start+lengths

    gt= small_pred!=large_pred

    defect_input = test_input.copy()
    for i in range(start,end):
        if defect_input[i]==gt[i]:
            defect_input[i] = abs(1-defect_input[i])
    print()
    np.save('./defect_input_upload_results.npy',defect_input)
        


if __name__=='__main__':
    # upload_with_input()
    get_nobalance_input()