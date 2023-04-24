# coding: utf-8
import argparse
import numpy as np


def original_data_para():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', default='../dataset/hz/', type=str)
    parser.add_argument('--data_file', default="matrix_in_30.npz")
    parser.add_argument('--day_index', default="day-index-201901.txt")
    parser.add_argument('--model', default='all_models/', type=str)
    parser.add_argument('--data_type', default='InOD', type=str, help="InOD, OutOD, In, Out")
    parser.add_argument('--input_types', default=['P', 'D', 'W'], type=list)
    parser.add_argument('--input_steps', default=[3, 0, 0], type=list, help="3-6")
    parser.add_argument('--data_mode', type=str, default="data1", help='data creat mode')
    return parser

# 所有数据处理data_model.py都会有的
def common_para(parser):
    parser.add_argument('--dataset_dir', default='../dataset/hz/', type=str)
    parser.add_argument('--N', default=80, type=int)
    parser.add_argument('--P', default=3, type=int)
    parser.add_argument('--train_ratio', type=float, default=0.7)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--data_log', default='data_log', type=str)
    parser.add_argument('--Null_Val', type=float, default=np.nan, help='value for missing data')

    parser.add_argument('--Batch_Size', type=int, default=32)
    parser.add_argument('--max_grad_norm', type=float, default=5.)
    parser.add_argument('--opt', type=str, default='adam')
    parser.add_argument('--max_epoch', type=int, default=1000, help='epoch to run')
    parser.add_argument('--patience', type=int, default=15, help='patience for early stop')
    parser.add_argument('--lr_decay_ratio', type=float, default=0.9)
    parser.add_argument('--steps', type=int, default=1, help="five learning rate")
    parser.add_argument('--min_learning_rate', type=float, default=2.0e-06)
    parser.add_argument('--base_lr', type=float, default=0.01,help='initial learning rate')
    parser.add_argument('--data_mode', type=str, default="data1", help='data creat mode')
    parser.add_argument('--continue_train', type=int, default=0, help='initial withou old model')
    parser.add_argument('--train_shuffle', type=int, default=0, help='train shuffle')
    parser.add_argument('--description', type=str, default="no description", help='parameter description or experience description')

    return parser





# GEML的data和main都会有的
def MVPF_main(parser):
    parser.add_argument('--model', default='MVPF', type=str)

    parser.add_argument('--data_type', default='InOD', type=str, help="InOD, OutOD, In, Out")
    parser.add_argument('--input_types', default=['P', 'D', 'W'], type=list)
    parser.add_argument('--input_steps', default=[3, 0, 0], type=list, help="3-6")
    parser.add_argument('--data_names', default="labels-samples_P-in_flow-od_out-out_flow", type=str)
    parser.add_argument('--output_type', default='matrix', type=str, help="sclar, vector, matrix")
    parser.add_argument('--Normalize', type=bool, default=True)
    parser.add_argument('--proportion', default=1.0, type=float)

    parser.add_argument('--GPU', type=int, default=2)
    parser.add_argument('--loss_type', default=1, type=int, help="1-single, 2-multi_man, 3-multi_auto")

    parser.add_argument('--train_log', default='train_log', type=str)
    parser.add_argument('--result_log', default='result_log', type=str)
    parser.add_argument('--data', default=1, type=int,help="produce data")
    parser.add_argument('--experiment', default=1, type=int,help="Do experiments")
    parser.add_argument('--Times', default=1, type=int, help="the number of experiment")
    return parser


# GEML模型参数
def MVPF(parser):
    parser.add_argument('--test_mode', default=0, type=int)

    parser.add_argument('--ave_time', default='mean.xlsx', type=str)

    parser.add_argument('--GCN_activations', default="tanh-tanh-tanh", type=str)
    parser.add_argument('--GCN_units', default="128-128-128", type=str)
    parser.add_argument('--Ks', default=[None,None], type=list)

    parser.add_argument('--RNN_units', default="128-128-128", type=str)
    parser.add_argument('--RNN_type', default="GRU", type=str)
    parser.add_argument('--D', default=4, type=int)

    return parser

