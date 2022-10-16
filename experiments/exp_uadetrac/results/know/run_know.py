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

def get_nobalance_know():
    small_pred = np.load(weak_pred_path)
    large_pred = np.load(strong_pred_path)

    test_input = np.load('./know_winlen60_upload_results.npy')

    tlens = len(test_input)

    start = 5000
    lengths  = 15000
    end = start+lengths

    gt= small_pred!=large_pred

    defect_input = test_input.copy()
    for i in range(start,end):
        if defect_input[i]==gt[i]:
            defect_input[i] = abs(1-defect_input[i])
    print()
    np.save('./defect_know_upload_results.npy',defect_input)
        


if __name__=='__main__':
    # upload_with_input()
    get_nobalance_know()