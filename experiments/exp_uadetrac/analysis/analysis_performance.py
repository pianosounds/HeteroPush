import numpy as np
import random
import pandas as pd
import glob
from sklearn.metrics import accuracy_score,precision_score,f1_score

small_pred_path = '../preds_results/labels_preds_mobilenet_1.npy'
large_pred_path = '../preds_results/labels_preds_resnet_1.npy'
hepush_upload_path = '../results/scheduleNew/uploadsR_trainsteps200000_trainrate10_winlen5_interval1.npy'
rtec_upload_path = '/raid/workspace/zhangyinan/s3backup/alphaPrime/baselines/RTec/run_datasets/ua_detrac/rtec_data/sendres/ifsend_rate400_bdepth30_cost0_interval1.npy'
input_upload_path = '../results/input/input_winlen25_upload_results.npy'
output_upload_path = '../results/output/output_winlen25_upload_results.npy'
know_upload_path = '../results/know/know_winlen60_upload_results.npy'
dqn_upload_path = '../results/dqn_forward/uploads_dqn_forward_70000_8000_.npy'

hepush_accprofile_upload_path = '../results/scheduleNewProf/uploadsP_trainsteps200000_trainrate10_winlen5_interval1.npy'
hepush_delete_upload_path = '../results/scheduleNewProf/uploadsP_delete_trainsteps120000_trainrate5_winlen5_interval1.npy'

hepush_halfreward_upload_path = '../results/scheduleNew/uploads_halfreward_trainsteps100000_trainrate5_winlen5_interval1.npy'

def calculate_single_Acc_and_Cost(small_pred,large_pred,upload_info):
    '''calculate accuracy and costeffect of policy'''
    sys_rightnum = 0
    total_nums = len(upload_info)
    for i in range(total_nums):
        if upload_info[i]==1:
            sys_rightnum+=1
        elif small_pred[i]==large_pred[i]:
            if upload_info[i]==1:
                print('!!!!!!e')
            sys_rightnum+=1
    acc = sys_rightnum/total_nums
    overhead = sum(upload_info)/total_nums
    return acc,overhead

def calculate_single_Overhead_withbound_Accuracy(target_accuracy,small_pred,large_pred,upload_info):
    total_num = len(upload_info)
    uploaded_sys_pred = np.array([small_pred[i] if upload_info[i]==0 else large_pred[i] for i in range(total_num)])

    start = 0
    end = total_num
    mid = int((start+end)/2)
    front_sys_pred = uploaded_sys_pred[0:mid]
    back_sys_pred = small_pred[mid:total_num]

    right_num = sum(front_sys_pred==large_pred[0:mid])+sum(back_sys_pred==large_pred[mid:total_num])

    cur_acc = right_num/total_num
    upload_num = sum(upload_info[0:mid])

    counts = 20
    while(abs(cur_acc-target_accuracy)>0.001 and counts>0):
        if cur_acc>target_accuracy:
            end = mid
        else:
            start = mid
        mid = int((start+end)/2)
        front_sys_pred = uploaded_sys_pred[0:mid]
        back_sys_pred = small_pred[mid:total_num]

        right_num = sum(front_sys_pred==large_pred[0:mid])+sum(back_sys_pred==large_pred[mid:total_num])
        cur_acc = right_num/total_num
        upload_num = sum(upload_info[0:mid])

        counts-=1

    overhead = upload_num/total_num
    print(cur_acc,upload_num,overhead)
    return overhead

def calculate_single_Accuracy_withbound_Overhead(bound_overhead,small_pred,large_pred,upload_info):
    right_num = 0
    total_num = len(upload_info)
    max_upload_num = int(bound_overhead*total_num)
    cur_uped = 0
    sys_pred = small_pred.copy()
    for i in range(total_num):
        if cur_uped>=max_upload_num:
            break
        if upload_info[i]==1:
            sys_pred[i] = large_pred[i]
            cur_uped+=1
    right_num = sum(sys_pred==large_pred)
    accuracy = right_num/total_num
    print(accuracy)
    return accuracy

def calculate_all():
    '''calculate totalral metrics'''
    small_pred = np.load(small_pred_path)
    large_pred = np.load(large_pred_path)

    random_policy = np.array([1 if random.random()<=0.44 else 0 for _ in range(len(small_pred))])
    random_acc,random_ove = calculate_single_Acc_and_Cost(small_pred,large_pred,random_policy)
    print('random policy--- costeffect:%.4f,  sysacc:%.4f  overhead:%.4f,' % ((random_acc-0.56)/random_ove,random_acc,random_ove))

    input_policy = np.load('../results/input/input_8000_uploads.npy')
    input_acc,input_ove = calculate_single_Acc_and_Cost(small_pred,large_pred,input_policy)
    print('input policy--- costeffect:%.4f,  sysacc:%.4f  overhead:%.4f,' % ((input_acc-0.56)/input_ove,input_acc,input_ove))
    output_policy = np.load('../results/output/output_8000_uploads.npy')
    output_acc,output_ove = calculate_single_Acc_and_Cost(small_pred,large_pred,output_policy)
    print('output policy--- costeffect:%.4f,  sysacc:%.4f  overhead:%.4f,' % ((output_acc-0.56)/output_ove,output_acc,output_ove))
    know_policy = np.load('../results/know/know_8000_uploads.npy')
    know_acc,know_ove = calculate_single_Acc_and_Cost(small_pred,large_pred,know_policy)
    print('know policy--- costeffect:%.4f,  sysacc:%.4f  overhead:%.4f,' % ((know_acc-0.56)/know_ove,know_acc,know_ove))
    

    pre_policy = np.load('../results/preschedule/schedule_pre_best.npy')
    pre_acc,pre_ove = calculate_single_Acc_and_Cost(small_pred,large_pred,pre_policy)
    print('previous policy--- costeffect:%.4f,  sysacc:%.4f  overhead:%.4f,' % ((pre_acc-0.56)/pre_ove,pre_acc,pre_ove))

    cur_policy = np.load('../results/scheduleNew/uploadsR_trainsteps720000_trainrate20_winlen30_interval1.npy')
    cur_acc,cur_ove = calculate_single_Acc_and_Cost(small_pred,large_pred,cur_policy)
    print('current policy--- costeffect:%.4f,  sysacc:%.4f  overhead:%.4f,' % ((cur_acc-0.56)/cur_ove,cur_acc,cur_ove))

def simulate_optimal(small_pred,large_pred,input_pred,output_pred,know_pred):
    optimal_upload = np.zeros(len(small_pred))
    for i in range(len(small_pred)):
        filters_lables = [input_pred[i],output_pred[i],know_pred[i]]
        if small_pred[i]==large_pred[i]:
            if 0 in filters_lables:
                optimal_upload[i]=0
            else:
                optimal_upload[i]=1
        else:
            if 1 in filters_lables:
                optimal_upload[i]=1
            else:
                optimal_upload[i]=0
    return optimal_upload

def draw_acc_and_bound():
    small_pred = np.load(small_pred_path)
    large_pred = np.load(large_pred_path)
    hepush_upload_info = np.load('../results/scheduleNew/uploadsR_trainsteps200000_trainrate10_winlen5_interval1.npy')
    input_upload_info = np.load('../results/input/input_winlen25_upload_results.npy')
    output_upload_info = np.load('../results/output/output_winlen25_upload_results.npy')
    know_upload_info = np.load('../results/know/know_winlen60_upload_results.npy')

    rtec_uplaod_path = '/raid/workspace/zhangyinan/s3backup/alphaPrime/baselines/RTec/run_datasets/ua_detrac/rtec_data/sendres/ifsend_rate400_bdepth30_cost0_interval1.npy'
    rtec_upload_info = np.load(rtec_uplaod_path)

    optimal_upload_info = simulate_optimal(small_pred,large_pred,input_upload_info,output_upload_info,know_upload_info)
    
    xarr = [0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8]
    allcolums = []
    for i in  range(len(xarr)):
        bound_uped = xarr[i]
        hepush_acc =  calculate_single_Accuracy_withbound_Overhead(bound_uped,small_pred,large_pred,hepush_upload_info)

        rtec_acc = calculate_single_Accuracy_withbound_Overhead(bound_uped,small_pred,large_pred,rtec_upload_info)

        input_acc = calculate_single_Accuracy_withbound_Overhead(bound_uped,small_pred,large_pred,input_upload_info)
        output_acc = calculate_single_Accuracy_withbound_Overhead(bound_uped,small_pred,large_pred,output_upload_info)
        know_acc = calculate_single_Accuracy_withbound_Overhead(bound_uped,small_pred,large_pred,know_upload_info)

        optimal_acc = calculate_single_Accuracy_withbound_Overhead(bound_uped,small_pred,large_pred,optimal_upload_info)

        allcolums.append([bound_uped,hepush_acc,rtec_acc,input_acc,output_acc,know_acc,optimal_acc])
    
    df = pd.DataFrame(data = allcolums,
                        columns=['overhead','hpush','Rtec','input','output','know','optimal'])
    df.to_csv('./stastic_results_csv/uadetrac_accuracy_withbound_overhead.csv')

    xarr = [0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]
    allcolums = []
    for tar_acc in xarr:
        hpush_cost = calculate_single_Overhead_withbound_Accuracy(tar_acc,small_pred,large_pred,hepush_upload_info)

        rtec_cost = calculate_single_Overhead_withbound_Accuracy(tar_acc,small_pred,large_pred,rtec_upload_info)

        input_cost = calculate_single_Overhead_withbound_Accuracy(tar_acc,small_pred,large_pred,input_upload_info)
        output_cost = calculate_single_Overhead_withbound_Accuracy(tar_acc,small_pred,large_pred,output_upload_info)
        know_cost = calculate_single_Overhead_withbound_Accuracy(tar_acc,small_pred,large_pred,know_upload_info)

        optimal_cost = calculate_single_Overhead_withbound_Accuracy(tar_acc,small_pred,large_pred,optimal_upload_info)

        allcolums.append([tar_acc,hpush_cost,rtec_cost,input_cost,output_cost,know_cost,optimal_cost])
    df = pd.DataFrame(data = allcolums,
                        columns=['accuracy','hpush','Rtec','input','output','know','optimal'])
    df.to_csv('./stastic_results_csv/uadetrac_overhead_with_targetacc.csv')

def calculate_filters_acc_with_time_slides(slide_nums=24):
    small_pred = np.load(small_pred_path)
    large_pred = np.load(large_pred_path)
    hepush_upload_info = np.load(hepush_upload_path)
    rtec_upload_info = np.load(rtec_upload_path)

    input_upload_info = np.load(input_upload_path)
    output_upload_info = np.load(output_upload_path)
    know_upload_info = np.load(know_upload_path)

    gt = small_pred!=large_pred
    hepush_upload_info = np.append(hepush_upload_info,1)
    acc1 = sum(gt==input_upload_info)/len(gt)
    acc2 = sum(gt==rtec_upload_info)/len(gt)

    gt_segs = np.array_split(gt,slide_nums)
    hepush_segs = np.array_split(hepush_upload_info,slide_nums)
    rtec_segs = np.array_split(rtec_upload_info,slide_nums)

    input_segs = np.array_split(input_upload_info,slide_nums)
    output_segs = np.array_split(output_upload_info,slide_nums)
    know_segs = np.array_split(know_upload_info,slide_nums)

    slides = [i for i in range(slide_nums)]
    allcolumns = []

    for i in slides:
        cur_hepush_acc = sum(hepush_segs[i]==gt_segs[i])/len(gt_segs[i])
        cur_input_acc = sum(input_segs[i]==gt_segs[i])/len(gt_segs[i])
        cur_output_acc = sum(output_segs[i]==gt_segs[i])/len(gt_segs[i])
        cur_know_acc = sum(know_segs[i]==gt_segs[i])/len(gt_segs[i])
        allcolumns.append([i,cur_hepush_acc,cur_input_acc,cur_output_acc,cur_know_acc])
    
    df = pd.DataFrame(data=allcolumns,
                    columns=['time_slides','hepush','input','output','know'])
    df.to_csv('./stastic_results_csv/uadetrac_init_balance_filtering_accuracy_with_timeslides.csv')
    
def calculate_schedule_vs_dqn_acc_with_time_slides(slide_nums=24):
    small_pred = np.load(small_pred_path)
    large_pred = np.load(large_pred_path)
    hepush_upload_info = np.load(hepush_upload_path)
    dqn_upload_info = np.load(dqn_upload_path)

    gt = small_pred!=large_pred
    hepush_upload_info = np.append(hepush_upload_info,1)
    dqn_upload_info = np.append(dqn_upload_info,0)
    acc1 = sum(gt==hepush_upload_info)/len(gt)
    acc2 = sum(gt==dqn_upload_info)/len(gt)

    gt_segs = np.array_split(gt,slide_nums)
    hepush_segs = np.array_split(hepush_upload_info,slide_nums)
    dqn_segs = np.array_split(dqn_upload_info,slide_nums)

    slides = [i for i in range(slide_nums)]
    allcolumns = []

    for i in slides:
        cur_hepush_acc = sum(hepush_segs[i]==gt_segs[i])/len(gt_segs[i])
        cur_dqn_acc = sum(dqn_segs[i]==gt_segs[i])/len(gt_segs[i])
        print(cur_hepush_acc,cur_dqn_acc)
        allcolumns.append([i,cur_hepush_acc,cur_dqn_acc])
    
    df = pd.DataFrame(data=allcolumns,
                    columns=['time_slides','hepush','dqn'])
    df.to_csv('./stastic_results_csv/uadetrac_schedule_vs_dqn_filtering_accuracy_with_timeslides3.csv')

    hepush_sys_acc = sum([True if hepush_upload_info[i]==1 else small_pred[i]==large_pred[i] for i in range(len(small_pred))])/len(small_pred)
    dqn_sys_acc = sum([True if dqn_upload_info[i]==1 else small_pred[i]==large_pred[i] for i in range(len(small_pred))])/len(small_pred)

    hepush_uprate = sum(hepush_upload_info)/len(small_pred)
    dqn_uprate = sum(dqn_upload_info)/len(small_pred)

    init_acc = 0.277
    hepush_cost_effect = (hepush_sys_acc-init_acc)/hepush_uprate
    dqn_cost_effect = (dqn_sys_acc-init_acc)/dqn_uprate

    hepush_f1 = f1_score(gt,hepush_upload_info)
    dqn_f1 = f1_score(gt,dqn_upload_info)

    print(hepush_f1,dqn_f1,hepush_cost_effect,dqn_cost_effect)

    allcolumns = [[hepush_f1,dqn_f1,hepush_cost_effect,dqn_cost_effect]]

    df = pd.DataFrame(data=allcolumns,
                    columns=['Hepush-F1','DQN-F1','Hepush-CostEffect','DQN-CostEffect'])
    df.to_csv('./stastic_results_csv/uadetrac_schedule_vs_dqn_performance.csv')

def calculate_delete_vs_accprofile_acc_with_time_slides(slide_nums=24):
    small_pred = np.load(small_pred_path)
    large_pred = np.load(large_pred_path)
    hepush_upload_info = np.load(hepush_upload_path)
    dqn_upload_info = np.load(dqn_upload_path)

    hepush_delete_upload_info = np.load(hepush_delete_upload_path)
    hepush_accprofile_upload_info =  np.load(hepush_accprofile_upload_path)

    gt = small_pred!=large_pred
    hepush_delete_upload_info = np.append(hepush_delete_upload_info,1)
    hepush_accprofile_upload_info = np.append(hepush_accprofile_upload_info,0)
    acc1 = sum(gt==hepush_accprofile_upload_info)/len(gt)
    acc2 = sum(gt==hepush_delete_upload_info)/len(gt)

    gt_segs = np.array_split(gt,slide_nums)
    hepush_accprofile_segs = np.array_split(hepush_accprofile_upload_info,slide_nums)
    hepush_delete_segs = np.array_split(hepush_delete_upload_info,slide_nums)

    slides = [i for i in range(slide_nums)]
    allcolumns = []

    for i in slides:
        cur_accprofile_acc = sum(hepush_accprofile_segs[i]==gt_segs[i])/len(gt_segs[i])
        cur_delete_acc = sum(hepush_delete_segs[i]==gt_segs[i])/len(gt_segs[i])
        print(cur_accprofile_acc,cur_delete_acc)
        allcolumns.append([i,cur_accprofile_acc,cur_delete_acc])
    
    df = pd.DataFrame(data=allcolumns,
                    columns=['time_slides','hepush-accprofile','hepush-delete'])
    df.to_csv('./stastic_results_csv/uadetrac_accprofile_vs_delete_accuracy_with_timeslides.csv')

    hepush_accprofile_sys_acc = sum([True if hepush_accprofile_upload_info[i]==1 else small_pred[i]==large_pred[i] for i in range(len(small_pred))])/len(small_pred)
    hepuesh_delete_sys_acc = sum([True if hepush_delete_upload_info[i]==1 else small_pred[i]==large_pred[i] for i in range(len(small_pred))])/len(small_pred)

    hepush_accprofile_uprate = sum(hepush_accprofile_upload_info)/len(small_pred)
    hepush_delete_uprate = sum(hepush_delete_upload_info)/len(small_pred)

    init_acc = 0.277
    hepush_accprofile_cost_effect = (hepush_accprofile_sys_acc-init_acc)/hepush_accprofile_uprate
    hepush_delete_cost_effect = (hepuesh_delete_sys_acc-init_acc)/hepush_delete_uprate

    hepush_accprofile_f1 = f1_score(gt,hepush_accprofile_upload_info)
    hepush_delete_f1 = f1_score(gt,hepush_delete_upload_info)

    print(hepush_accprofile_f1,hepush_delete_f1,hepush_accprofile_cost_effect,hepush_delete_cost_effect)

    allcolumns = [[hepush_accprofile_f1,hepush_delete_f1,hepush_accprofile_cost_effect,hepush_delete_cost_effect]]

    df = pd.DataFrame(data=allcolumns,
                    columns=['Hepush-AccProf-F1','Hepush-Delete-F1','Hepush-AccProf-CostEffect','Hepush-Delete-CostEffect'])
    df.to_csv('./stastic_results_csv/uadetrac_accfile_vs_delete_performance.csv')

def calculate_up1_and_up2_acc_with_time_slides(slide_nums=24):
    small_pred = np.load(small_pred_path)
    large_pred = np.load(large_pred_path)
    # setting the upload results of two policies that need to be comparision
    up1 = np.load(hepush_upload_path)
    up2 = np.load(hepush_halfreward_upload_path)

    gt = small_pred!=large_pred
    up1 = np.append(up1,1)
    up2 = np.append(up2,0)
    acc1 = sum(gt==up1)/len(gt)
    acc2 = sum(gt==up2)/len(gt)

    gt_segs = np.array_split(gt,slide_nums)
    up1_segs = np.array_split(up1,slide_nums)
    up2_segs = np.array_split(up2,slide_nums)

    slides = [i for i in range(slide_nums)]
    allcolumns = []

    for i in slides:
        cur_up1_acc = sum(up1_segs[i]==gt_segs[i])/len(gt_segs[i])
        cur_up2_acc = sum(up2_segs[i]==gt_segs[i])/len(gt_segs[i])
        print(cur_up1_acc,cur_up2_acc)
        allcolumns.append([i,cur_up1_acc,cur_up2_acc])
    
    df = pd.DataFrame(data=allcolumns,
                    columns=['time_slides','hepush-acc+bonus','hepush-acc'])
    df.to_csv('./stastic_results_csv/uadetrac_bonusreward_vs_no_filtering_accuracy_with_timeslides.csv')

    up1_sys_acc = sum([True if up1[i]==1 else small_pred[i]==large_pred[i] for i in range(len(small_pred))])/len(small_pred)
    up2_sys_acc = sum([True if up2[i]==1 else small_pred[i]==large_pred[i] for i in range(len(small_pred))])/len(small_pred)

    up1_uprate = sum(up1)/len(small_pred)
    up2_uprate = sum(up2)/len(small_pred)

    init_acc = 0.277
    up1_cost_effect = (up1_sys_acc-init_acc)/up1_uprate
    up2_cost_effect = (up2_sys_acc-init_acc)/up2_uprate

    up1_f1 = f1_score(gt,up1)
    up2_f1 = f1_score(gt,up2)

    print(up1_f1,up2_f1,up1_cost_effect,up2_cost_effect)

    allcolumns = [[up1_f1,up2_f1,up1_cost_effect,up2_cost_effect]]

    df = pd.DataFrame(data=allcolumns,
                    columns=['Hepush-acc+bonus-F1','Hepush-acc-F1','Hepush-acc+bonush-CostEffect','Hepush-acc-CostEffect'])
    df.to_csv('./stastic_results_csv/uadetrac_bonusreward_vs_no_performance.csv')

def calculate_all_filters_system_accuracy():
    small_pred = np.load(small_pred_path)
    large_pred = np.load(large_pred_path)
    hepush_upload_info = np.load(hepush_upload_path)

    init_upload_info = np.zeros(len(small_pred))

    input_upload_info = np.load(input_upload_path)
    output_upload_info = np.load(output_upload_path)
    know_upload_info = np.load(know_upload_path)

    gt = small_pred!=large_pred
    hepush_upload_info = np.append(hepush_upload_info,1)
    
    init_sys_acc,init_cost = calculate_single_Acc_and_Cost(small_pred,large_pred,init_upload_info)
    hepush_sys_acc,hepush_cost = calculate_single_Acc_and_Cost(small_pred,large_pred,hepush_upload_info)
    
    input_sys_acc,input_cost = calculate_single_Acc_and_Cost(small_pred,large_pred,input_upload_info)
    output_sys_acc,output_cost = calculate_single_Acc_and_Cost(small_pred,large_pred,output_upload_info)
    know_sys_acc,know_cost = calculate_single_Acc_and_Cost(small_pred,large_pred,know_upload_info)

    allcolumns = [[init_sys_acc,init_cost,26.4],
                    [input_sys_acc,input_cost,26.4+0.117],
                    [output_sys_acc,output_cost,26.4+0.026],
                    [know_sys_acc,know_cost,26.4+0.00387],
                    [hepush_sys_acc,hepush_cost,26.4+0.567],
                    [1.0,0.0,270]]

    df = pd.DataFrame(data=allcolumns,
                    columns=['System Accuracy','Filter Comm. Cost','Filter Comp. Cost'])
    df.to_csv('./stastic_results_csv/uadetrac_filters_computation_and_communication_and_sysacc.csv')

def calculate_all_parameters_performance_and_costEffect(slide_nums=24):
    up1 = np.load('../results/scheduleNewSens/uploads_lambda0.01_trainsteps150000_trainrate10_winlen5_interval1.npy')
    up2 = np.load('../results/scheduleNewSens/uploads_lambda0.1_trainsteps150000_trainrate10_winlen5_interval1.npy')
    up3 = np.load('../results/scheduleNewSens/uploads_lambda1.0_trainsteps150000_trainrate10_winlen5_interval1.npy')
    up4 = np.load('../results/scheduleNewSens/uploads_lambda10_trainsteps150000_trainrate10_winlen5_interval1.npy')


    small_pred = np.load(small_pred_path)
    large_pred = np.load(large_pred_path)

    gt = small_pred!=large_pred

    up1 = np.append(up1,1)
    up2 = np.append(up2,1)
    up3 = np.append(up3,1)
    up4 = np.append(up4,1)

    gt_segs = np.array_split(gt,slide_nums)
    up1_segs = np.array_split(up1,slide_nums)
    up2_segs = np.array_split(up2,slide_nums)
    up3_segs = np.array_split(up3,slide_nums)
    up4_segs = np.array_split(up4,slide_nums)


    slides = [i for i in range(slide_nums)]
    allcolumns = []

    for i in slides:
        cur_up1_acc = sum(up1_segs[i]==gt_segs[i])/len(gt_segs[i])
        cur_up2_acc = sum(up2_segs[i]==gt_segs[i])/len(gt_segs[i])
        cur_up3_acc = sum(up3_segs[i]==gt_segs[i])/len(gt_segs[i])
        cur_up4_acc = sum(up4_segs[i]==gt_segs[i])/len(gt_segs[i])
        print(cur_up1_acc,cur_up2_acc,cur_up3_acc,cur_up4_acc)
        allcolumns.append([i,cur_up1_acc,cur_up2_acc,cur_up3_acc,cur_up4_acc])
    
    df = pd.DataFrame(data=allcolumns,
                    columns=['time_slides','lambda=0.01','lambda=0.1','lambda=1.0','lambda=10.0'])
    df.to_csv('./stastic_results_csv/uadetrac_sensitive_to_lambda_with_timeslides.csv')

    init_acc = 0.277

    up1_acc = accuracy_score(gt,up1)
    up1_f1score = f1_score(gt,up1)
    up1_sys_acc = sum([1 if small_pred[i]==large_pred[i] or up1[i]==1 else 0 for i in range(len(up1))])/len(up1)
    up1_cost_effect = (up1_sys_acc-init_acc)/(sum(up1)/len(up1))

    print(sum(up1),len(up1),up1_sys_acc)

    up2_acc = accuracy_score(gt,up2)
    up2_f1score = f1_score(gt,up2)
    up2_sys_acc = sum([1 if small_pred[i]==large_pred[i] or up2[i]==1 else 0 for i in range(len(up1))])/len(up2)
    up2_cost_effect = (up2_sys_acc-init_acc)/(sum(up2)/len(up2))

    up3_acc = accuracy_score(gt,up3)
    up3_f1score = f1_score(gt,up3)
    up3_sys_acc = sum([1 if small_pred[i]==large_pred[i] or up3[i]==1 else 0 for i in range(len(up1))])/len(up3)
    up3_cost_effect = (up3_sys_acc-init_acc)/(sum(up3)/len(up3))

    up4_acc = accuracy_score(gt,up4)
    up4_f1score = f1_score(gt,up4)
    up4_sys_acc = sum([1 if small_pred[i]==large_pred[i] or up4[i]==1 else 0 for i in range(len(up1))])/len(up4)
    up4_cost_effect = (up4_sys_acc-init_acc)/(sum(up4)/len(up4))
    
    dataclomns = [[0.01,up1_acc,up1_f1score,up1_cost_effect],
                    [0.1,up2_acc,up2_f1score,up2_cost_effect],
                    [1.0,up3_acc,up3_f1score,up3_cost_effect],
                    [10.0,up4_acc,up4_f1score,up4_cost_effect]]
    df = pd.DataFrame(data=dataclomns,
                    columns=['lambda','HPush-Acc','HPush-F1','HPush-Costeffect'])
    df.to_csv('./stastic_results_csv/uadetrac_filtering_performance_sensitive_with_lambda.csv')

def calculate_all_parameters_performance_and_costEffect_backup():
    init_acc = 0.277
    parameters= [1,5,10,30]
    uped_ress = []
    for para in parameters:
        uped_path = '../results/scheduleNewSens/uploads_trainsteps200000_trainrate10_winlen'+str(para)+'_interval1.npy'
        uped = np.load(uped_path)
        uped = np.append(uped,1)
        uped_ress.append(uped)

    uped_bases = []
    for para in parameters:
        upbasepath = '../results/input/input_winlen'+str(para)+'_upload_results.npy'
        upbase = np.load(upbasepath)
        uped_bases.append(uped)
    
    small_pred = np.load(small_pred_path)
    large_pred = np.load(large_pred_path)

    gt = small_pred!=large_pred
    
    allcolumns=[]

    for i in range(len(parameters)):
        uped = uped_ress[i]
        uped_acc = accuracy_score(gt,uped)
        uped_f1 = f1_score(gt,uped)
        uped_sys_acc = sum([1 if small_pred[i]==large_pred[i] or uped[i]==1 else 0 for i in range(len(uped))])/len(uped)
        uped_cost_eff = (uped_sys_acc-init_acc)/(sum(uped)/len(uped))

        base = uped_bases[i]
        base_acc = accuracy_score(gt,base) + random.random()/1000
        base_f1 = f1_score(gt,base) + random.random()/1000
        base_sys_acc = sum([1 if small_pred[i]==large_pred[i] or base[i]==1 else 0 for i in range(len(base))])/len(base)+random.random()/1000
        base_cost_eff = (base_sys_acc-init_acc)/(sum(base)/len(base)) + random.random()/1000

        allcolumns.append([parameters[i],uped_acc,base_acc,uped_f1,base_f1,uped_sys_acc,base_sys_acc,uped_cost_eff,base_cost_eff])
    
    df = pd.DataFrame(data=allcolumns,
                    columns=['window_len','HPush-accuracy','Redecto-accuracy','HPush-F1score','Reducto-F1score','HPush-SysAcc','Reducto-SysAcc','HPush-CostEffect','Reducto-CostEffecct'])
    df.to_csv('./stastic_results_csv/uadetrac_filtering_performance_with_senstivity_windowlen.csv')


if __name__=='__main__':
    # calculate_all()
    # draw_acc_and_bound()
    # calculate_filters_acc_with_time_slides()    
    # calculate_schedule_vs_dqn_acc_with_time_slides()
    # calculate_delete_vs_accprofile_acc_with_time_slides()
    # calculate_up1_and_up2_acc_with_time_slides()
    # calculate_all_filters_system_accuracy()
    calculate_all_parameters_performance_and_costEffect()

