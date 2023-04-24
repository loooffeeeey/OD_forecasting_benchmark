import os
import sys
import argparse
import time
import random

import torch
import torch.nn as nn
import torch.autograd.profiler as profiler

stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
from dgl.dataloading import GraphDataLoader
sys.stderr.close()
sys.stderr = stderr

from utils import Logger, batch2device, plot_grad_flow, evalMetrics, genMetricsResStorage, aggrMetricsRes, wrapMetricsRes
from RSODPDataSet import RSODPDataSet
from model import Gallat, GallatExt, GallatExtFull, AR, LSTNet, GCRN, GEML

import Config
if Config.CHECK_GRADS:
    torch.autograd.set_detect_anomaly(True)


def batch2res(batch, device, *args):
    net, tune, ref_ext = args[-1]
    record, record_GD, record_GCRN, query, target_G, target_D = batch['record'], batch['record_GD'], batch['record_GCRN'], batch['query'], batch['target_G'], batch['target_D']
    if device:
        record, record_GD, record_GCRN, query, target_G, target_D = batch2device(record, record_GD, record_GCRN, query, target_G, target_D, device)

    if net.__class__.__name__ in ['AR', 'LSTNet']:
        res_D, res_G = net(record_GD)
    elif net.__class__.__name__ == 'GCRN':
        res_D, res_G = net(record_GCRN)
    elif net.__class__.__name__ == 'GEML':
        res_D, res_G = net(record['Sp'])
    else:
        res_D, res_G = net(record, record_GD, query, predict_G=True, ref_extent=ref_ext)

    return res_D, res_G, target_D, target_G


def loadRefAR(ref_AR_path, device):
    if ref_AR_path == 'None':
        return None
    if not os.path.exists(ref_AR_path):
        sys.stderr.write('[TRAIN] The referenced AR model path %s is invalid!\n' % ref_AR_path)
        exit(-55)
    refAR = torch.load(ref_AR_path, map_location=device)
    if refAR.__class__.__name__ != 'AR':
        sys.stderr.write(
            '[TRAIN] The referenced AR model is not an AR model (got %s)!\n' % refAR.__class__.__name__)
        exit(-555)
    return refAR


def train(lr=Config.LEARNING_RATE_DEFAULT, bs=Config.BATCH_SIZE_DEFAULT, ep=Config.MAX_EPOCHS_DEFAULT,
          eval_freq=Config.EVAL_FREQ_DEFAULT, opt=Config.OPTIMIZER_DEFAULT, num_workers=Config.WORKERS_DEFAULT,
          use_gpu=True, gpu_id=Config.GPU_ID_DEFAULT,
          data_dir=Config.DATA_DIR_DEFAULT, logr=Logger(activate=False),
          unify_FB=False, mix_FB=False,
          model=Config.NETWORK_DEFAULT, ref_AR_path=Config.REF_AR_DEFAULT,
          model_save_dir=Config.MODEL_SAVE_DIR_DEFAULT, train_type=Config.TRAIN_TYPE_DEFAULT,
          metrics_threshold=torch.Tensor([0]), total_H=Config.DATA_TOTAL_H, start_H=Config.DATA_START_H,
          hidden_dim=Config.HIDDEN_DIM_DEFAULT, feat_dim=Config.FEAT_DIM_DEFAULT, query_dim=Config.QUERY_DIM_DEFAULT,
          retrain_model_path=Config.RETRAIN_MODEL_PATH_DEFAULT, loss_function=Config.LOSS_FUNC_DEFAULT,
          tune=True, ref_ext=Config.REF_EXTENT):
    # CUDA if possible
    device = torch.device('cuda:%d' % gpu_id if (use_gpu and torch.cuda.is_available()) else 'cpu')
    logr.log('> device: {}\n'.format(device))

    # Load DataSet
    logr.log('> Loading DataSet from {}\n'.format(data_dir))
    dataset = RSODPDataSet(data_dir,
                           his_rec_num=Config.HISTORICAL_RECORDS_NUM_DEFAULT,
                           time_slot_endurance=Config.TIME_SLOT_ENDURANCE_DEFAULT,
                           total_H=total_H, start_at=start_H,
                           unify_FB=unify_FB, mix_FB=mix_FB)
    trainloader = GraphDataLoader(dataset.train_set, batch_size=bs, shuffle=True, num_workers=num_workers)
    validloader = GraphDataLoader(dataset.valid_set, batch_size=bs, shuffle=False, num_workers=num_workers)
    logr.log('> Total Hours: {}, starting from {}\n'.format(dataset.total_H, dataset.start_at))
    logr.log('> Unify FB Graphs: {}, Mix FB Graphs: {}\n'.format(unify_FB, mix_FB))
    logr.log('> Training batches: {}, Validation batches: {}\n'.format(len(trainloader), len(validloader)))

    # Initialize the Model
    predict_G = (train_type != 'pretrain')
    task = 'OD' if predict_G else 'Demand'
    refAR = loadRefAR(ref_AR_path, device)
    net = Gallat(feat_dim=feat_dim, query_dim=query_dim, hidden_dim=hidden_dim)
    if train_type == 'retrain':
        logr.log('> Loading the Pretrained Model: {}, Train type = {}\n'.format(retrain_model_path, train_type))
        net = torch.load(retrain_model_path, map_location=device)
    else:
        logr.log('> Initializing the Training Model: {}, Train type = {}\n'.format(model, train_type))
        if model == 'Gallat':
            net = Gallat(feat_dim=feat_dim, query_dim=query_dim, hidden_dim=hidden_dim, num_dim=3, tune=tune, ref_AR=refAR)
        elif model == 'GallatExt':
            net = GallatExt(feat_dim=feat_dim, query_dim=query_dim, hidden_dim=hidden_dim, num_heads=Config.NUM_HEADS_DEFAULT, num_dim=3, tune=tune, ref_AR=refAR)
        elif model == 'GallatExtFull':
            net = GallatExtFull(feat_dim=feat_dim, query_dim=query_dim, hidden_dim=hidden_dim, num_heads=Config.NUM_HEADS_DEFAULT, num_dim=3, tune=tune, ref_AR=refAR)
        elif model == 'AR':
            net = AR(p=Config.HISTORICAL_RECORDS_NUM_DEFAULT)
        elif model == 'LSTNet':
            if refAR is None:
                sys.stderr.write('[TRAIN] The referenced AR model path %s is invalid for LSTNet!\n' % ref_AR_path)
                exit(-55)
            net = LSTNet(p=Config.HISTORICAL_RECORDS_NUM_DEFAULT, refAR=refAR)
        elif model == 'GCRN':
            net = GCRN(num_nodes=dataset.grid_info['gridNum'], hidden_dim=hidden_dim)
        elif model == 'GEML':
            net = GEML(feat_dim=feat_dim, hidden_dim=hidden_dim)

    logr.log('> Model Structure:\n{}\n'.format(net))

    # Select Optimizer
    logr.log('> Constructing the Optimizer: {}\n'.format(opt))
    optimizer = torch.optim.Adam(net.parameters(), lr, weight_decay=Config.WEIGHT_DECAY_DEFAULT)  # Default: Adam + L2 Norm
    if opt == 'ADAM':
        optimizer = torch.optim.Adam(net.parameters(), lr, weight_decay=Config.WEIGHT_DECAY_DEFAULT)    # Adam + L2 Norm

    # Loss Function
    logr.log('> Using {} as the Loss Function.\n'.format(loss_function))
    criterion_D = nn.SmoothL1Loss(beta=Config.SMOOTH_L1_LOSS_BETA_DEFAULT)
    criterion_G = nn.SmoothL1Loss(beta=Config.SMOOTH_L1_LOSS_BETA_DEFAULT)
    if loss_function == 'SmoothL1Loss':
        criterion_D = nn.SmoothL1Loss(beta=Config.SMOOTH_L1_LOSS_BETA_DEFAULT)
        criterion_G = nn.SmoothL1Loss(beta=Config.SMOOTH_L1_LOSS_BETA_DEFAULT)
    elif loss_function == 'MSELoss':
        criterion_D = nn.MSELoss()
        criterion_G = nn.MSELoss()

    if device:
        net.to(device)
        logr.log('> Model sent to {}\n'.format(device))

    # Referenced Extent
    if device:
        ref_ext = torch.Tensor([ref_ext]).to(device)

    # Model Saving Directory
    if not os.path.isdir(model_save_dir):
        os.mkdir(model_save_dir)

    # Metrics
    metrics_threshold_val = metrics_threshold.item()
    if device:
        metrics_threshold = metrics_threshold.to(device)

    # Summarize Info
    logr.log('\nlearning_rate = {}, epochs = {}, num_workers = {}\n'.format(lr, ep, num_workers))
    logr.log('eval_freq = {}, batch_size = {}, optimizer = {}\n'.format(eval_freq, bs, opt))
    if model in Config.NETWORKS_TUNABLE:
        logr.log('tune = %s%s\n' % (str(tune), ", use_AR=%s, ref_extent = %.2f" % (net.ref_AR, ref_ext.item()) if tune else ""))
    if model in Config.MULTI_HEAD_ATT_APPLICABLE:
        logr.log('num_heads = %d\n' % Config.NUM_HEADS_DEFAULT)
    if predict_G:
        logr.log('Demand task ~ %.2f%%, OD task ~ %.2f%%\n' % (Config.D_PERCENTAGE_DEFAULT * 100, Config.G_PERCENTAGE_DEFAULT * 100))

    # Start Training
    logr.log('\nStart Training!\n')
    logr.log('------------------------------------------------------------------------\n')

    min_eval_loss = float('inf')

    for epoch_i in range(ep):
        # train one round
        net.train()
        train_loss = 0
        train_metrics_res = genMetricsResStorage(num_metrics_threshold=1, tasks=[task])
        time_start_train = time.time()
        for i, batch in enumerate(trainloader):
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            record, record_GD, record_GCRN, query, target_G, target_D = batch['record'], batch['record_GD'], batch['record_GCRN'], batch['query'], batch['target_G'], batch['target_D']
            if device:
                record, record_GD, record_GCRN, query, target_G, target_D = batch2device(record, record_GD, record_GCRN, query, target_G, target_D, device)

            # Avoid exploding gradients
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=Config.MAX_NORM_DEFAULT)

            optimizer.zero_grad()

            if Config.PROFILE:
                with profiler.profile(profile_memory=True, use_cuda=True) as prof:
                    with profiler.record_function('model_inference'):
                        if model == 'AR':
                            res_D, res_G = net(record_GD)
                        elif model == 'GCRN':
                            res_D, res_G = net(record_GCRN)
                        elif model == 'GEML':
                            res_D, res_G = net(record['Sp'])
                        else:
                            res_D, res_G = net(record, record_GD, query, predict_G=predict_G, ref_extent=ref_ext)  # if pretrain, res_G = None
                logr.log(prof.key_averages().table(sort_by="cuda_time_total"))
                exit(100)

            if model in ['AR', 'LSTNet']:
                res_D, res_G = net(record_GD)
            elif model == 'GCRN':
                res_D, res_G = net(record_GCRN)
            elif model == 'GEML':
                res_D, res_G = net(record['Sp'])
            else:
                res_D, res_G = net(record, record_GD, query, predict_G=predict_G, ref_extent=ref_ext)  # if pretrain, res_G = None

            loss = (criterion_D(res_D, target_D) * Config.D_PERCENTAGE_DEFAULT + criterion_G(res_G, target_G) * Config.G_PERCENTAGE_DEFAULT) if predict_G else criterion_D(res_D, target_D)

            loss.backward()

            if Config.CHECK_GRADS:
                plot_grad_flow(net.named_parameters())

            optimizer.step()

            # Analysis
            with torch.no_grad():
                train_loss += loss.item()
                train_metrics_res = aggrMetricsRes(train_metrics_res, [metrics_threshold], 1,
                                                   res_D, target_D, res_G, target_G)

            if Config.TRAIN_JUST_ONE_BATCH:     # DEBUG
                if i == 0:
                    break

        # Analysis after one training round in the epoch
        train_loss /= len(trainloader)
        train_metrics_res = wrapMetricsRes(train_metrics_res)
        time_end_train = time.time()
        total_train_time = (time_end_train - time_start_train)
        train_time_per_sample = total_train_time / len(dataset.train_set)
        logr.log('Training Round %d: loss = %.6f, time_cost = %.4f sec (%.4f sec per sample), RMSE-%d = %.4f, MAPE-%d = %.4f, MAE-%d = %.4f\n' %
                 (epoch_i + 1, train_loss, total_train_time, train_time_per_sample, metrics_threshold_val, train_metrics_res[task]['RMSE']['val'], metrics_threshold_val, train_metrics_res[task]['MAPE']['val'], metrics_threshold_val, train_metrics_res[task]['MAE']['val']))

        # eval_freq: Evaluate on validation set
        if (epoch_i + 1) % eval_freq == 0:
            net.eval()
            val_loss_total = 0
            valid_metrics_res = genMetricsResStorage(num_metrics_threshold=1, tasks=[task])
            with torch.no_grad():
                for j, val_batch in enumerate(validloader):
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                    val_record, val_record_GD, val_record_GCRN, val_query, val_target_G, val_target_D = val_batch['record'], val_batch['record_GD'], val_batch['record_GCRN'], val_batch['query'], val_batch['target_G'], val_batch['target_D']
                    if device:
                        val_record, val_record_GD, val_record_GCRN, val_query, val_target_G, val_target_D = batch2device(val_record, val_record_GD, val_record_GCRN, val_query, val_target_G, val_target_D, device)

                    if model in ['AR', 'LSTNet']:
                        val_res_D, val_res_G = net(val_record_GD)
                    elif model == 'GCRN':
                        val_res_D, val_res_G = net(val_record_GCRN)
                    elif model == 'GEML':
                        val_res_D, val_res_G = net(val_record['Sp'])
                    else:
                        val_res_D, val_res_G = net(val_record, val_record_GD, val_query, predict_G=predict_G, ref_extent=ref_ext)

                    val_loss = criterion_D(val_res_D, val_target_D) * Config.D_PERCENTAGE_DEFAULT + criterion_G(val_res_G, val_target_G) * Config.G_PERCENTAGE_DEFAULT if predict_G else criterion_D(val_res_D, val_target_D)

                    val_loss_total += val_loss.item()
                    valid_metrics_res = aggrMetricsRes(valid_metrics_res, [metrics_threshold], 1,
                                                       val_res_D, val_target_D, val_res_G, val_target_G)

                val_loss_total /= len(validloader)
                valid_metrics_res = wrapMetricsRes(valid_metrics_res)
                logr.log('!!! Validation : loss = %.6f, RMSE-%d = %.4f, MAPE-%d = %.4f, MAE-%d = %.4f\n' %
                         (val_loss_total, metrics_threshold_val, valid_metrics_res[task]['RMSE']['val'], metrics_threshold_val, valid_metrics_res[task]['MAPE']['val'], metrics_threshold_val, valid_metrics_res[task]['MAE']['val']))

                if epoch_i >= 10 and val_loss_total < min_eval_loss:
                    min_eval_loss = val_loss_total
                    model_name = os.path.join(model_save_dir, '{}.pth'.format(logr.time_tag))
                    torch.save(net, model_name)
                    logr.log('Model: {} has been saved since it achieves smaller loss.\n'.format(model_name))

        if Config.TRAIN_JUST_ONE_ROUND:
            if epoch_i == 0:    # DEBUG
                break

    # End Training
    logr.log('> Training finished.\n')


def evaluate(model_name, bs=Config.BATCH_SIZE_DEFAULT, num_workers=Config.WORKERS_DEFAULT,
             use_gpu=True, gpu_id=Config.GPU_ID_DEFAULT,
             data_dir=Config.DATA_DIR_DEFAULT, logr=Logger(activate=False),
             unify_FB=False, mix_FB=False,
             total_H=Config.DATA_TOTAL_H, start_H=Config.DATA_START_H,
             tune=True, ref_ext=Config.REF_EXTENT):
    """
        Evaluate using saved best model (Note that this is a Test API)
        1. Re-evaluate on the validation set
        2. Re-evaluate on the test set
        The evaluation metrics include RMSE, MAPE, MAE
    """
    # CUDA if needed
    device = torch.device('cuda:%d' % gpu_id if (use_gpu and torch.cuda.is_available()) else 'cpu')
    logr.log('> device: {}\n'.format(device))

    # Load Model
    logr.log('> Loading {}\n'.format(model_name))
    net = torch.load(model_name, map_location=device)
    logr.log('> Model Structure:\n{}\n'.format(net))

    if device:
        net.to(device)
        logr.log('> Model sent to {}\n'.format(device))

    # Load DataSet
    logr.log('> Loading DataSet from {}\n'.format(data_dir))
    dataset = RSODPDataSet(data_dir,
                           his_rec_num=Config.HISTORICAL_RECORDS_NUM_DEFAULT,
                           time_slot_endurance=Config.TIME_SLOT_ENDURANCE_DEFAULT,
                           total_H=total_H, start_at=start_H,
                           unify_FB=unify_FB, mix_FB=mix_FB)
    validloader = GraphDataLoader(dataset.valid_set, batch_size=bs, shuffle=False, num_workers=num_workers)
    testloader = GraphDataLoader(dataset.test_set, batch_size=bs, shuffle=False, num_workers=num_workers)
    logr.log('> Total Hours: {}, starting from {}\n'.format(dataset.total_H, dataset.start_at))
    logr.log('> Unify FB Graphs: {}, Mix FB Graphs: {}\n'.format(unify_FB, mix_FB))
    logr.log('> Validation batches: {}, Test batches: {}\n'.format(len(validloader), len(testloader)))

    # Referenced Extent
    if device:
        ref_ext = torch.Tensor([ref_ext]).to(device)

    # Log Info
    if net.__class__.__name__ in Config.NETWORKS_TUNABLE:
        logr.log('tune = %s%s\n' % (str(tune), ", ref_extent = %.2f" % ref_ext.item() if tune else ""))
    if net.__class__.__name__ in Config.MULTI_HEAD_ATT_APPLICABLE:
        logr.log('num_heads = %d\n' % net.num_heads)

    net.eval()
    # 1.
    evalMetrics(validloader, 'Validation', batch2res, device, logr, net, tune, ref_ext)

    # 2.
    evalMetrics(testloader, 'Test', batch2res, device, logr, net, tune, ref_ext)

    # End Evaluation
    logr.log('> Evaluation finished.\n')


if __name__ == '__main__':
    """ 
        Usage Example:
        python Trainer.py -dr data/ny2016_0101to0331/ -th 1064 -ts 1 -c 4 -m train -tt pretrain -net Gallat -me 200 -bs 5 -re 0.2
        python Trainer.py -dr data/ny2016_0101to0331/ -th 1064 -ts 1 -c 4 -m train -tt retrain -r res/Gallat_pretrain/20210514_07_17_13.pth -me 100 -bs 5 -re 0.2
        python Trainer.py -dr data/ny2016_0101to0331/ -th 1064 -ts 1 -c 4 -m eval -e res/Gallat_normal/20210515_16_47_01.pth -bs 5 -re 0.2
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-lr', '--learning_rate', type=float, default=Config.LEARNING_RATE_DEFAULT, help='Learning rate, default = {}'.format(Config.LEARNING_RATE_DEFAULT))
    parser.add_argument('-me', '--max_epochs', type=int, default=Config.MAX_EPOCHS_DEFAULT, help='Number of epochs to run the trainer, default = {}'.format(Config.MAX_EPOCHS_DEFAULT))
    parser.add_argument('-ef', '--eval_freq', type=int, default=Config.EVAL_FREQ_DEFAULT, help='Frequency of evaluation on the validation set, default = {}'.format(Config.EVAL_FREQ_DEFAULT))
    parser.add_argument('-bs', '--batch_size', type=int, default=Config.BATCH_SIZE_DEFAULT, help='Size of a minibatch, default = {}'.format(Config.BATCH_SIZE_DEFAULT))
    parser.add_argument('-opt', '--optimizer', type=str, default=Config.OPTIMIZER_DEFAULT, help='Optimizer to be used [ADAM], default = {}'.format(Config.OPTIMIZER_DEFAULT))
    parser.add_argument('-dr', '--data_dir', type=str, default=Config.DATA_DIR_DEFAULT, help='Root directory of the input data, default = {}'.format(Config.DATA_DIR_DEFAULT))
    parser.add_argument('-th', '--hours', type=int, default=Config.DATA_TOTAL_H, help='Specify the number of hours for data, default = {}'.format(Config.DATA_TOTAL_H))
    parser.add_argument('-ts', '--start_hour', type=int, default=Config.DATA_START_H, help='Specify the starting hour for data, default = {}'.format(Config.DATA_START_H))
    parser.add_argument('-u', '--ufb', type=int, default=Config.UNIFY_FB_DEFAULT, help='Specify whether to unify FB graphs, default = {}'.format(Config.UNIFY_FB_DEFAULT))
    parser.add_argument('-mfb', '--mfb', type=int, default=Config.MIX_FB_DEFAULT, help='Specify whether to mix FB graphs, default = {}'.format(Config.MIX_FB_DEFAULT))
    parser.add_argument('-ld', '--log_dir', type=str, default=Config.LOG_DIR_DEFAULT, help='Specify where to create a log file. If log files are not wanted, value will be None'.format(Config.LOG_DIR_DEFAULT))
    parser.add_argument('-c', '--cores', type=int, default=Config.WORKERS_DEFAULT, help='number of workers (cores used), default = {}'.format(Config.WORKERS_DEFAULT))
    parser.add_argument('-gpu', '--gpu', type=int, default=Config.USE_GPU_DEFAULT, help='Specify whether to use GPU, default = {}'.format(Config.USE_GPU_DEFAULT))
    parser.add_argument('-gid', '--gpu_id', type=int, default=Config.GPU_ID_DEFAULT, help='Specify which GPU to use, default = {}'.format(Config.GPU_ID_DEFAULT))
    parser.add_argument('-net', '--network', type=str, default=Config.NETWORK_DEFAULT,  help='Specify which model/network to use, default = {}'.format(Config.NETWORK_DEFAULT))
    parser.add_argument('-rar', '--ref_ar', type=str, default=Config.REF_AR_DEFAULT, help='Specify the location of the saved AR model to be used as a reference for training LSTNet, default = {}'.format(Config.REF_AR_DEFAULT))
    parser.add_argument('-m', '--mode', type=str, default=Config.MODE_DEFAULT, help='Specify which mode the discriminator runs in (train, eval, trainNeval), default = {}'.format(Config.MODE_DEFAULT))
    parser.add_argument('-e', '--eval', type=str, default=Config.EVAL_DEFAULT, help='Specify the location of saved network to be loaded for evaluation, default = {}'.format(Config.EVAL_DEFAULT))
    parser.add_argument('-md', '--model_save_dir', type=str, default=Config.MODEL_SAVE_DIR_DEFAULT, help='Specify the location of network to be saved, default = {}'.format(Config.MODEL_SAVE_DIR_DEFAULT))
    parser.add_argument('-tt', '--train_type', type=str, default=Config.TRAIN_TYPE_DEFAULT, help='Specify train mode [normal, pretrain, retrain], default = {}'.format(Config.TRAIN_TYPE_DEFAULT))
    parser.add_argument('-mt', '--metrics_threshold', type=int, default=0, help='Specify the metrics threshold, default = {}'.format(0))
    parser.add_argument('-hd', '--hidden_dim', type=int, default=Config.HIDDEN_DIM_DEFAULT, help='Specify the hidden dimension, default = {}'.format(Config.HIDDEN_DIM_DEFAULT))
    parser.add_argument('-fd', '--feature_dim', type=int, default=Config.FEAT_DIM_DEFAULT, help='Specify the feature dimension, default = {}'.format(Config.FEAT_DIM_DEFAULT))
    parser.add_argument('-qd', '--query_dim', type=int, default=Config.QUERY_DIM_DEFAULT, help='Specify the query dimension, default = {}'.format(Config.QUERY_DIM_DEFAULT))
    parser.add_argument('-r', '--retrain_model_path', type=str, default=Config.RETRAIN_MODEL_PATH_DEFAULT, help='Specify the location of the model to be retrained if train type is retrain, default = {}'.format(Config.RETRAIN_MODEL_PATH_DEFAULT))
    parser.add_argument('-lf', '--loss_function', type=str, default=Config.LOSS_FUNC_DEFAULT, help='Specify which loss function to use, default = {}'.format(Config.LOSS_FUNC_DEFAULT))
    parser.add_argument('-t', '--tune', type=int, default=Config.TUNE_DEFAULT, help='Specify whether to tune in the transferring layer of the model, default = {}'.format(Config.TUNE_DEFAULT))
    parser.add_argument('-re', '--ref_extent', type=float, default=Config.REF_EXTENT, help='The extent of referenced value in the tuning block of Transferring Layer, default = {}'.format(Config.REF_EXTENT))
    parser.add_argument('-tag', '--tag', type=str, default=Config.TAG_DEFAULT, help='Name tag for the model, default = {}'.format(Config.TAG_DEFAULT))

    FLAGS, unparsed = parser.parse_known_args()

    # Starts a log file in the specified directory
    logger = Logger(activate=True, logging_folder=FLAGS.log_dir, comment='%s_%s' % (FLAGS.tag, FLAGS.mode)) \
        if FLAGS.log_dir else Logger(activate=False)

    # Controls reproducibility
    if Config.RAND_SEED:
        random.seed(Config.RAND_SEED)
        torch.manual_seed(Config.RAND_SEED)
        logger.log('> Seed: %d\n' % Config.RAND_SEED)

    working_mode = FLAGS.mode
    if working_mode == 'train':
        train(lr=FLAGS.learning_rate, bs=FLAGS.batch_size, ep=FLAGS.max_epochs,
              eval_freq=FLAGS.eval_freq, opt=FLAGS.optimizer, num_workers=FLAGS.cores,
              use_gpu=(FLAGS.gpu == 1), gpu_id=FLAGS.gpu_id,
              data_dir=FLAGS.data_dir, logr=logger,
              unify_FB=(FLAGS.ufb == 1), mix_FB=(FLAGS.mfb == 1),
              model=FLAGS.network, ref_AR_path=FLAGS.ref_ar,
              model_save_dir=FLAGS.model_save_dir, train_type=FLAGS.train_type,
              metrics_threshold=torch.Tensor([FLAGS.metrics_threshold]),
              total_H=FLAGS.hours, start_H=FLAGS.start_hour, hidden_dim=FLAGS.hidden_dim,
              feat_dim=FLAGS.feature_dim, query_dim=FLAGS.query_dim,
              retrain_model_path=FLAGS.retrain_model_path, loss_function=FLAGS.loss_function,
              tune=(FLAGS.tune == 1), ref_ext=FLAGS.ref_extent)
        logger.close()
    elif working_mode == 'eval':
        eval_file = FLAGS.eval
        # Abnormal: file not found
        if (not eval_file) or (not os.path.isfile(eval_file)):
            sys.stderr.write('File for evaluation not found, please check!\n')
            logger.close()
            exit(-1)
        # Normal
        evaluate(eval_file, bs=FLAGS.batch_size, num_workers=FLAGS.cores,
                 use_gpu=(FLAGS.gpu == 1), gpu_id=FLAGS.gpu_id,
                 data_dir=FLAGS.data_dir, logr=logger, total_H=FLAGS.hours, start_H=FLAGS.start_hour,
                 unify_FB=(FLAGS.ufb == 1), mix_FB=(FLAGS.mfb == 1),
                 tune=(FLAGS.tune == 1), ref_ext=FLAGS.ref_extent)
        logger.close()
    elif working_mode == 'trainNeval':
        # First train then eval
        train(lr=FLAGS.learning_rate, bs=FLAGS.batch_size, ep=FLAGS.max_epochs,
              eval_freq=FLAGS.eval_freq, opt=FLAGS.optimizer, num_workers=FLAGS.cores,
              use_gpu=(FLAGS.gpu == 1), gpu_id=FLAGS.gpu_id,
              data_dir=FLAGS.data_dir, logr=logger,
              unify_FB=(FLAGS.ufb == 1), mix_FB=(FLAGS.mfb == 1),
              model=FLAGS.network, ref_AR_path=FLAGS.ref_ar,
              model_save_dir=FLAGS.model_save_dir, train_type=FLAGS.train_type,
              metrics_threshold=torch.Tensor([FLAGS.metrics_threshold]),
              total_H=FLAGS.hours, start_H=FLAGS.start_hour, hidden_dim=FLAGS.hidden_dim,
              feat_dim=FLAGS.feature_dim, query_dim=FLAGS.query_dim,
              retrain_model_path=FLAGS.retrain_model_path, loss_function=FLAGS.loss_function,
              tune=(FLAGS.tune == 1), ref_ext=FLAGS.ref_extent)

        saved_model_path = os.path.join(Config.MODEL_SAVE_DIR_DEFAULT, '%s.pth' % logger.time_tag)
        logger.log('\n')

        evaluate(saved_model_path, bs=FLAGS.batch_size, num_workers=FLAGS.cores,
                 use_gpu=(FLAGS.gpu == 1), gpu_id=FLAGS.gpu_id,
                 data_dir=FLAGS.data_dir, logr=logger, total_H=FLAGS.hours, start_H=FLAGS.start_hour,
                 unify_FB=(FLAGS.ufb == 1), mix_FB=(FLAGS.mfb == 1),
                 tune=(FLAGS.tune == 1), ref_ext=FLAGS.ref_extent)

        logger.close()
    else:
        sys.stderr.write('Please specify the running mode (train/eval/trainNeval)\n')
        logger.close()
        exit(-2)
