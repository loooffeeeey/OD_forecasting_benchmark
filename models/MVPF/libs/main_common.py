# conding=utf-8
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
import pandas as pd
import os
import numpy as np
import time
from sklearn import linear_model
from libs import utils, metrics, data_common
import datetime
# from trmf import Model, train
import tqdm


def main(args):
    for i in range(args.Times):
        # 产生数据
        if args.data:
            data_common.data_process(args)
        # 实验
        if args.experiment:
            experiment(args)
            tf.reset_default_graph()




########################################  Nerual Network #####################################################


def get_feed_dicts(data, placeholders, shuffle=0):
    num_batch = data['arr_0'].shape[0]

    feed_dicts = []
    if shuffle:
        per = list(np.random.permutation(num_batch))  # 随机划分
    else:
        per = range(num_batch)

    for j in per:
        feed_dict = {}
        for i in range(len(placeholders)):
            feed_dict[placeholders[i]] = data['arr_%s' % i][j, ...]
        feed_dicts.append(feed_dict)
    return feed_dicts


def choose_model(args):
    if args.model == "MVPF":
        import models.model_MVPF as model
    else:
        raise ValueError
    return model


def experiment(args):
    model = choose_model(args)  # 选择模型
    log_dir, save_dir, data_dir, dir_name = utils.create_dir_PDW(args)  # 按输入模式，产生文件夹
    log_res = utils.create_log(args, args.result_log)  # 结果log
    log = utils.create_log(args, args.train_log)  # 训练log
    utils.log_string(log, str(args)[10: -1])  # 打印超参数
    model_file = save_dir + '/'

    # 装载数据
    data, mean, std = data_common.load_data(args, log, data_dir, yes=args.Normalize)

    if args.model in["MVPF"]:
        path = os.path.join(data_dir, 'graph_geo.npz')
        graph_geo = np.load(path)['graph'].astype(np.float32)
        df = pd.read_excel(os.path.join(args.dataset_dir, 'original', args.ave_time), index_col=0)
        ave_time = df.values
        ave_time = ave_time.astype(np.float32)
        path = os.path.join(args.dataset_dir, 'original',  'embedding_lda.npz')
        lda_feature = np.load(path)['feature'].astype(np.float32)

        path = os.path.join(args.dataset_dir, 'original', 'SE.npz')
        SE = np.load(path)["SE"].astype(np.float32)
        utils.log_string(log, 'SE:  {}'.format(SE.shape))

        features = lda_feature


    utils.log_string(log, 'Data Loaded Finish...')

    # 模型编译
    utils.log_string(log, 'compiling model...')
    P, N = args.P, int(args.N)
    F_in, F_out = data[1]['arr_1'].shape[-1], N
    utils.log_string(log, "P-%s, F_in-%s, F_out-%s" % (P, F_in, F_out))


    if args.model == "MVPF":
        placeholders = model.placeholder(P, N, F_in, F_out)
        labels, samples, flow_in, od_out, flow_out = placeholders
        # graph_sem = tf.identity(graph_sem, name='graph_sem')
        preds, In_preds, Out_preds = model.Model(args, mean, std, samples, od_out,flow_in, flow_out,ave_time,graph_geo,SE,features)


    # 改名字
    preds = tf.identity(preds, name='preds')


    Preds = [preds, In_preds, Out_preds]
    In_labels, Out_labels = tf.reduce_sum(labels, axis=-1), tf.reduce_sum(labels, axis=-2)
    Labels = [labels, In_labels, Out_labels]

    # sys.exit()
    if args.loss_type == 1:  # 单个目标
        loss = metrics.masked_mse_tf(preds, labels, null_val=args.Null_Val)  # 损失
    elif args.loss_type == 2:  # 多个目标-自动权重
        loss = metrics.total_loss(args, Preds, Labels)
    elif args.loss_type == 3:  # 多个目标-人为权重
        loss = metrics.total_loss_W(args, Preds, Labels)

    lr, new_lr, lr_update, train_op = optimization(args, loss)  # 优化
    utils.print_parameters(log)  # 打印参数
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)  # 保存模型

    # 配置GPU
    sess = GPU(args.GPU)



    if args.continue_train:
        val_loss_min, Epoch = restore(log, sess, model_file, saver)
    else:
        # 初始化模型
        val_loss_min = np.inf
        Epoch = 0
        sess.run(tf.global_variables_initializer())
    wait = 0
    step = 0

    utils.log_string(log, "initializer successfully")

    utils.log_string(log, '**** training model ****')



    Message = ''
    save_loss = [[], [], []]
    epoch = Epoch
    while (epoch < args.max_epoch):
        # for epoch in range(Epoch, args.max_epoch):
        # 降低学习率
        if wait >= args.patience:
            val_loss_min, epoch = restore(log, sess, model_file, saver)
            step += 1
            wait = 0
            New_Lr = max(args.min_learning_rate, args.base_lr * (args.lr_decay_ratio ** step))
            sess.run(lr_update, feed_dict={new_lr: New_Lr})
            # 删除多余的loss
            if epoch > args.patience:
                for k in range(len(save_loss)):
                    save_loss[k] = save_loss[k][:-args.patience]
            if step > args.steps:
                utils.log_string(log, 'early stop at epoch: %04d' % (epoch))
                break

        # 打印当前时间/训练轮数/lr
        utils.log_string(log,
                         '%s | epoch: %04d/%d, lr: %.4f' %
                         (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), epoch, args.max_epoch, sess.run(lr)))

        # 计算训练集/验证集/测试集损失
        types = ['train', 'val', 'test']
        results = []
        for i in range(len(types)):
            feed_dicts = get_feed_dicts(data[i], placeholders, args.train_shuffle)
            result, run_message = caculation(args, log, sess, data_dir, feed_dicts, labels, preds, train_op, loss,
                                         type=types[i])
            if i == 2:  # 获取测试集的性能指标
                message_cur = run_message
            results.append(result)
        epoch_message = "loss=> train:%.4f val:%.4f test:%.4f time=> train:%.1f val:%.1f test:%.1f" % (
        results[0][0], results[1][0], results[2][0], results[0][1], results[1][1], results[2][1])
        utils.log_string(log, epoch_message)

        # 存储损失
        save_loss[0].append(results[0][0])
        save_loss[1].append(results[1][0])
        save_loss[2].append(results[2][0])


        val_loss = results[1][0]

        wait, val_loss_min, Message = update_val_loss(log, sess, saver, model_file, epoch, wait, val_loss, val_loss_min,
                                                      message_cur, Message)
        epoch += 1


    # 存储好损失
    path = os.path.join(data_dir, 'losses.npz')
    np.savez_compressed(path, np.array(save_loss))
    utils.log_string(log_res, Message)
    utils.log_string(log, Message)
    log.close()
    log_res.close()
    sess.close()


'''
配置优化器/学习率/剪裁梯度/反向更新/
'''


def optimization(args, loss):
    lr = tf.Variable(tf.constant_initializer(args.base_lr)(shape=[]),
                     dtype=tf.float32, trainable=False, name='learning_rate')  # (F, F1)

    global_step = tf.train.get_or_create_global_step()
    # lr = tf.get_variable('learning_rate', initializer=tf.constant(args.base_lr), trainable=False)
    new_lr = tf.placeholder(tf.float32, shape=(), name='new_learning_rate')
    lr_update = tf.assign(lr, new_lr)
    if args.opt == 'adam':
        optimizer = tf.train.AdamOptimizer(lr, epsilon=1e-3)
    elif args.opt == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(lr)
    elif args.opt == 'amsgrad':
        from libs.AMSGrad import AMSGrad
        optimizer = AMSGrad(lr)
    elif args.opt == 'delta':
        optimizer = tf.train.AdadeltaOptimizer(lr, epsilon=1e-3)
    elif args.opt == 'gradDAO':
        optimizer = tf.train.AdagradDAOptimizer(lr,global_step)
    elif args.opt == 'grad':
        optimizer = tf.train.AdagradOptimizer(lr)
    # elif args.opt == 'momentum':
    #     optimizer = tf.train.MomentumOptimizer(lr)
    elif args.opt == 'proximal':
        optimizer = tf.train.ProximalAdagradOptimizer(lr)
    elif args.opt == 'rms':
        optimizer = tf.train.RMSPropOptimizer(lr)
    # elif args.opt == 'sync':
    #     optimizer = tf.train.SyncReplicasOptimizer(lr)
    else:
        print("Optimizer error")
        raise ValueError

    # clip
    tvars = tf.trainable_variables()
    grads = tf.gradients(loss, tvars)
    grads, _ = tf.clip_by_global_norm(grads, args.max_grad_norm)
    train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step, name='train_op')
    return lr, new_lr, lr_update, train_op


'''
配置GPU
'''


def GPU(number):
    # print("i use gpu: ",number)
    # GPU configuration
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ["CUDA_VISIBLE_DEVICES"] = str(number)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    return sess


'''
恢复模型或初始化模型
'''


def restore(log, sess, model_file, saver):
    ckpt = tf.train.get_checkpoint_state(model_file)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        Epoch = int(ckpt.model_checkpoint_path.split('-')[-1]) + 1
        val_loss_min = np.load(model_file + 'val_loss_min.npz')['loss']
        message = "restore successfully, path:%s, Epoch:%d" % (ckpt.model_checkpoint_path, Epoch)
        utils.log_string(log, message)
    else:
        val_loss_min = np.inf
        Epoch = 0
        sess.run(tf.global_variables_initializer())
        utils.log_string(log, "initializer successfully")
    return val_loss_min, Epoch


'''
更新损失最小值，存储最好的模型，存储最小损失，npz，'loss'
'''


def update_val_loss(log, sess, saver, model_file, epoch, wait, loss, val_loss_min, message_cur, Message):
    # choose best test_loss
    if loss < val_loss_min:
        wait = 0
        val_loss_min = loss
        saver.save(sess, model_file, epoch)
        Message = message_cur
        np.savez(model_file + 'val_loss_min.npz', loss=val_loss_min)
        utils.log_string(log, "save %02d" % epoch)

        # graph = tf.get_default_graph()  # 获取当前默认计算图
        # w0 = graph.get_tensor_by_name("weight:0")
        # print(sess.run(w0))
        # w00 = graph.get_tensor_by_name("weight0:0")
        # print(sess.run(w00))
        # w01 = graph.get_tensor_by_name("weight1:0")
        # print(sess.run(w01))
        # w02 = graph.get_tensor_by_name("weight2:0")
        # print(sess.run(w02))

    else:
        wait += 1
    return wait, val_loss_min, Message


'''
查看预测的结果
'''


def pred_label(log, test_pred, test_label, data_dir):
    pred = np.reshape(test_pred, [-1, test_pred.shape[-1]])
    label = np.reshape(test_label, [-1, test_label.shape[-1]])
    diff = pred - label
    save(diff, data_dir, name='diff')
    save(pred, data_dir, name='pred')
    save(label, data_dir, name='label')
    utils.log_string(log, "save succesfully")


def save(values, data_dir, name):
    path = os.path.join(data_dir, '%s.xlsx' % name)
    df = pd.DataFrame(values)
    df.to_excel(path)


'''
计算一个epoch，训练时反向更新，测试时进行预测
'''


def caculation(args, log, sess, data_dir, feed_dicts, labels, preds, train_op, loss, type="train"):
    start = time.time()
    loss_all = []
    preds_all = []
    labels_all = []
    message_res = ''
    for feed_dict in feed_dicts:
        if type == "train":
            sess.run([train_op], feed_dict=feed_dict)
            # loss_list = sess.run([Losses], feed_dict=feed_dict)
        batch_loss = sess.run([loss], feed_dict=feed_dict)
        loss_all.append(batch_loss)
        if type == "test":
            batch_labels, batch_preds = sess.run([labels, preds], feed_dict=feed_dict)
            preds_all.append(batch_preds)
            labels_all.append(batch_labels)
    loss_mean = np.mean(loss_all)
    Time = time.time() - start

    if type == "test":
        preds_all = np.stack(preds_all, axis=0)
        preds_all = np.round(preds_all).astype(np.float32)
        labels_all = np.stack(labels_all, axis=0)
        mae, rmse, wmape, smape = metrics.calculate_metrics(preds_all, labels_all, null_val=args.Null_Val)
        message = "Test=> MAE:{:.4f} RMSE:{:.4f} WMAPE:{:.4f} SMAPE:{:.4f}".format(mae, rmse, wmape, smape)
        message_res = "MAE\t%.4f\tRMSE\t%.4f\tWMAPE\t%.4f\tSMAPE\t%.4f" % (mae, rmse, wmape, smape)
        utils.log_string(log, message)
        # 查看预测效果
        # pred_label(log, preds_all, labels_all, data_dir)
        # sys.exit()
    return [loss_mean, Time], message_res

