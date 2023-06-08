import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

'''
功能：计算两个多维数组之间的差距

把缺失值屏蔽掉，只计算非缺失值的平均loss
但OD中的0.0不代表缺失值，是真实数据，不能把它去掉
但0.0若作为真实值，计算MAPE时会作为分母，从而导致计算出错，因此计算MAPE的时候应该屏蔽掉这些值
'''

def masked_mae_tf(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~tf.is_nan(labels)
    else:
        mask = tf.not_equal(labels, null_val)
    mask = tf.cast(mask, tf.float32)
    mask /= tf.reduce_mean(mask)
    mask = tf.where(tf.is_nan(mask), tf.zeros_like(mask), mask)
    loss = tf.abs(tf.subtract(preds, labels))
    loss = loss * mask
    loss = tf.where(tf.is_nan(loss), tf.zeros_like(loss), loss)
    return tf.reduce_mean(loss)

def masked_mse_tf(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~tf.is_nan(labels)
    else:
        mask = tf.not_equal(labels, null_val)
    mask = tf.cast(mask, tf.float32)
    mask /= tf.reduce_mean(mask)
    mask = tf.where(tf.is_nan(mask), tf.zeros_like(mask), mask)
    loss = tf.square(tf.subtract(preds, labels))
    loss = loss * mask
    loss = tf.where(tf.is_nan(loss), tf.zeros_like(loss), loss)
    return tf.reduce_mean(loss)

def masked_rmse_tf(preds, labels, null_val=np.nan):
    return tf.sqrt(masked_mse_tf(preds=preds, labels=labels, null_val=null_val))

def masked_mae_np(preds, labels, null_val=np.nan,top_n_mask=None):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
            # print(labels.shape)
            # num = np.sum((~mask).astype('float32'))
            # print("null_val",type(null_val), null_val, num)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mae = np.abs(np.subtract(preds, labels)).astype('float32')
        mae = np.nan_to_num(mae * mask)
        if top_n_mask is not None:
            mae = mae*top_n_mask
        return np.mean(mae)

def masked_mape_np(preds, labels, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        # 屏蔽掉真实值是0.0的样本
        mask_zero = np.not_equal(labels, 0.0).astype('float32')
        mask = mask.astype('float32')
        mask = mask * mask_zero
        mask /= np.mean(mask)
        mape = np.abs(np.divide(np.subtract(preds, labels).astype('float32'), labels))
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape)

def masked_mse_np(preds, labels, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mse = np.square(np.subtract(preds, labels)).astype('float32')
        mse = np.nan_to_num(mse * mask)
        return np.mean(mse)

def masked_rmse_np(preds, labels, null_val=np.nan):
    return np.sqrt(masked_mse_np(preds=preds, labels=labels, null_val=null_val))

def masked_wmape1_np(preds, labels, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        weights = labels / np.sum(labels)
        mape = np.abs(np.divide(np.subtract(preds, labels).astype('float32'), labels))
        wmape = mape * weights
        wmape = np.nan_to_num(wmape * mask)
        wmape = np.sum(wmape)
        return wmape

def masked_wmape_np(preds, labels, null_val=np.nan,top_n_mask=None):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        diff = np.abs(np.subtract(preds, labels)).astype('float32')
        diff = np.nan_to_num(diff * mask)
        if top_n_mask is not None:
            diff = diff*top_n_mask
            SUM = np.sum(labels*top_n_mask)
        else:
            SUM = np.sum(labels)

        DIFF = np.sum(diff)
        wmape = DIFF / SUM
        # if top_n_mask is not None:
        #     wmape = wmape*top_n_mask
        return wmape

def masked_smape_np(preds, labels, null_val=np.nan,c=1.0):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        diff = np.subtract(preds, labels).astype('float32')
        sum = (np.abs(preds)+ np.abs(labels))*0.5 + c
        smape = np.abs(np.divide(diff, sum))
        smape = np.nan_to_num(mask * smape)
        smape = np.mean(smape)
        return smape

def calculate_metrics(preds, labels, null_val=np.nan,top_n_mask=None):
    mae = masked_mae_np(preds=preds, labels=labels, null_val=null_val,top_n_mask=top_n_mask)
    rmse = masked_rmse_np(preds=preds, labels=labels, null_val=null_val)
    #mape = masked_mape_np(preds=preds, labels=labels, null_val=null_val)
    wmape = masked_wmape_np(preds, labels, null_val=np.nan,top_n_mask=top_n_mask)
    smape = masked_smape_np(preds, labels, null_val=np.nan, c=1.0)
    # print(mae, rmse, wmape, smape)
    return mae, rmse, wmape, smape

def total_loss(args, preds, labels):
    num = len(preds)
    N = preds[0].shape[-1].value
    # loss_0 = masked_mse_tf(preds[0], labels[0], null_val=args.Null_Val)  # 损失
    # loss_1 = masked_mse_tf(preds[1], labels[1], null_val=args.Null_Val)/(N**2)  # 损失
    # loss_2 = masked_mse_tf(preds[2], labels[2], null_val=args.Null_Val)/(N**2)  # 损失
    # Loss = loss_0 * args.weights[0] + loss_1 * args.weights[1] + loss_2 * args.weights[2]
    Loss = tf.constant(0.0, dtype=tf.float32)
    for i in range(num):
        loss = masked_mse_tf(preds[i], labels[i], null_val=args.Null_Val)  # 损失
        if i!=0:
            loss = loss / (N**2)
        Loss += loss*args.weights[i]
    # return Loss,[loss_0,loss_1,loss_2]
    return Loss

def total_loss_W(args, preds, labels):
    N = preds[0].shape[-1].value
    num = len(preds)
    weights = []
    for i in range(num):
        w = tf.Variable(tf.glorot_uniform_initializer()(shape=()),
                        dtype=tf.float32, trainable=True, name='weight')  # (F1,)
        weights.append(w)
    Loss = tf.constant(0.0, dtype=tf.float32)
    for i in range(num):
        loss = masked_mse_tf(preds[i], labels[i], null_val=args.Null_Val)  # 损失
        if i!=0:
            loss = loss / (N**2)
        Loss += loss*tf.abs(weights[i])
    return Loss


