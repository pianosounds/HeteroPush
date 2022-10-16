# HeteroPush
The implementation of "HeteroPush: Communication-Efficient Video Analytics by Scheduling Heterogeneous Filters"

## Overview
![HeteroPush](/heteropush.png "Workflow of HeteroPush")

**Workflow of HeteroPush:** Image frames arrive to end devices continuously, there is a lightweight inference model deployed on end devices and give the inference results. Then, there are heterogeneous filters on the end devices and predict whether the current frame need to be filtered. Then Hpsh scheduler select one of them to get the final filtering policy.

## Train & Test
We have get the experiments on 2 dataset: ***/experiments/***. 

***/experiments/exp_dataset/pred_results*** including inference results and ***/experiments/exp_dataset/results*** including init filtering policy of heterogeneous filters.

To train a new scheduler, you just need to set up the parameters, train&test options and configs, run:

```
  python run_schedule.py
```
Then, the scheduler results and final filtering policy will be predicted and saved to ***/experiments/exp_dataset/results/run_schedule***.

## DQN based

We alse provide the method to filtering frames directly with DQN, similar to scheduler, you just need set up the parameters, train&test options and configs and run:

```
  python run_dqn_forward_profile.py
```

## Prepare Your Own Dataset

We also support you can use HeteroPush with you own dataset. You just need to create a new experiment directory like our exp-dir. And using your own filtering results of your heterogeneous filters.

