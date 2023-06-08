import os
import argparse
import matplotlib.pyplot as plt


LOG_FN_DEFAULT = 'dc_dataset/RefGaaRN_trainNeval_20220330_17_46_22.log'
OUT_DIR_DEFAULT = './'


def path2FileNameWithoutExt(path):
    """
    get file name without extension from path
    :param path: file path
    :return: file name without extension
    """
    return os.path.splitext(os.path.basename(path))[0]


def trainLog2LossCurve(logfn=LOG_FN_DEFAULT, outdir=OUT_DIR_DEFAULT):
    if not os.path.isfile(logfn):
        print('{} is not a valid file.'.format(logfn))
        exit(-1)

    if not os.path.isdir(outdir):
        print('{} is not a valid directory.'.format(outdir))
        exit(-2)

    x_epoch = []
    y_loss_train = []
    train_time_list = []

    print('Analyzing log file: {}'.format(logfn))
    f = open(logfn, 'r')
    lines = f.readlines()
    for line in lines:
        if not line.startswith('Training Round'):
            continue
        items = line.strip().split(sep=' ')

        epoch = int(items[2][:-1])
        x_epoch.append(epoch)

        loss = float(items[5][:-1])
        y_loss_train.append(loss)

        train_time = float(items[10][1:])
        train_time_list.append(train_time)

    # Count average TTpS
    avgTTpS = sum(train_time_list) / len(train_time_list)
    print('Average TTpS: %.4f sec' % avgTTpS)

    # Plot training loss curve
    print('Plotting loss curve.')
    plt.plot(x_epoch, y_loss_train, c='purple', label='Train Loss', alpha=0.8)
    plt.title('Epoch - Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    # plt.show()
    figpath = os.path.join(outdir, '{}.png'.format(path2FileNameWithoutExt(logfn)))
    plt.savefig(figpath)
    print('Loss curve saved to {}'.format(figpath))

    print('All analysis tasks finished.')


if __name__ == '__main__':
    """ 
        Usage Example:
        python LossCurveRenderer.py -i dc_dataset/RefGaaRN_trainNeval_20220330_17_46_22.log -o ./
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--in_log', type=str, default=LOG_FN_DEFAULT, help='The path of the input log file, default = {}'.format(LOG_FN_DEFAULT))
    parser.add_argument('-od', '--out_dir', type=str, default=OUT_DIR_DEFAULT, help='The directory to store the output picture, default = {}'.format(OUT_DIR_DEFAULT))

    FLAGS, unparsed = parser.parse_known_args()

    trainLog2LossCurve(logfn=FLAGS.in_log)
