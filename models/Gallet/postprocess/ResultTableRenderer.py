import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))   # Enables Config to be loaded
import argparse
from Config import METRICS_FOR_WHAT, METRICS_NAME, EVAL_METRICS_THRESHOLD_SET, OUR_MODEL, MODELS_TO_EXAMINE

LOG_IN_DIR_DEFAULT = './dc_dataset/'
DS_TAG_DEFAULT = 'dataset'


def path2FileNameWithoutExt(path):
    """
    get file name without extension from path
    :param path: file path
    :return: file name without extension
    """
    return os.path.splitext(os.path.basename(path))[0]


def extractFilesFromDir(in_dir=LOG_IN_DIR_DEFAULT):
    if not os.path.isdir(in_dir):
        print('{} is not a valid directory.'.format(in_dir))
        exit(-1)

    return next(os.walk(in_dir), (None, None, []))[2]  # [] if no file


def constructEmptyRecords():
    records = {}    # records[metrics][task][threshold] = value
    for cur_metrics in METRICS_NAME:
        records[cur_metrics] = {}
        for cur_task in METRICS_FOR_WHAT:
            records[cur_metrics][cur_task] = {}
            for cur_threshold in EVAL_METRICS_THRESHOLD_SET:
                records[cur_metrics][cur_task][str(cur_threshold)] = -1
    return records


def extractRecFromLog(file_path):
    if not os.path.isfile(file_path):
        print('{} is not a valid file.'.format(file_path))
        exit(-2)

    records = constructEmptyRecords()   # records[metrics][task][threshold] = value

    with open(file_path) as f:
        # Skip to the test results
        while True:
            line = f.readline().strip()
            # Scan when test results are here
            if 'Metrics Evaluations for Test Set' in line:
                break

        # Scan results for tasks sequentially
        cur_task = None
        for _ in range(len(METRICS_FOR_WHAT)):
            line = f.readline().strip()
            # pin-point a task
            for task in METRICS_FOR_WHAT:
                if task in line:
                    cur_task = task
                    break

            # Read all metrics
            for _ in range(len(METRICS_NAME)):
                items = f.readline().strip().split(', ')
                cur_metrics = None
                # pin-point a metrics
                for metrics in METRICS_NAME:
                    if metrics in items[0]:
                        cur_metrics = metrics
                        break
                for item in items:  # item is like "RMSE-0 = 28.5953"
                    blocks = item.split(' ')
                    cur_threshold = None
                    # pin-point a threshold
                    for threshold in EVAL_METRICS_THRESHOLD_SET:
                        if str(threshold) in blocks[0]:
                            cur_threshold = str(threshold)
                            break
                    # Store the value!
                    records[cur_metrics][cur_task][cur_threshold] = blocks[-1]

    return records


def summarizeRecInDir(in_dir=LOG_IN_DIR_DEFAULT):
    file_names = extractFilesFromDir(in_dir)
    t_records = {}      # t_records[model][metrics][task][threshold] = value
    for file_name in file_names:
        model_name = path2FileNameWithoutExt(file_name).split('_')[0]
        model_records = extractRecFromLog(os.path.join(in_dir, file_name))
        t_records[model_name] = model_records
    return t_records


# Latex format
def renderTable(t_records: dict, ds_tag: str):
    metrics_tables = {}
    num_models = len(t_records)
    for metrics in METRICS_NAME:
        # Table Meta Front
        render_str = '\\begin{table}\n' + \
                     '\\caption{%s results for the models}\n' % metrics + \
                     '\\label{tab:results%s}\n' % metrics + \
                     '\\centering\n' + \
                     '\\begin{tabular}{| c | c || c | c | c |}\n'
        # Table Head
        render_str += '\\hline\n' + \
                      '\\multirow{2}{*}{Task} & \\multirow{2}{*}{Model} & \\multicolumn{%d}{c|}{%s} \\\\\n' % (len(EVAL_METRICS_THRESHOLD_SET), ds_tag) + \
                      '\\cline{3-%d}\n' % (2 + len(EVAL_METRICS_THRESHOLD_SET)) + \
                      '& & ' + ' & '.join(['%s-%d' % (metrics, threshold) for threshold in EVAL_METRICS_THRESHOLD_SET]) + ' \\\\\n' + \
                      '\\hline\n'
        # Table Content
        for task in METRICS_FOR_WHAT:
            start = True
            for model_set in MODELS_TO_EXAMINE:
                for model in model_set:
                    if model not in t_records:
                        continue
                    cur_str = ''
                    if start:
                        cur_str += '\\multirow{%d}{*}{%s} & ' % (num_models, task)
                        start = False
                    else:
                        if model == OUR_MODEL:
                            cur_str += '\\cline{2-%d}\n' % (2 + len(EVAL_METRICS_THRESHOLD_SET))
                        cur_str += '& '
                    cur_str += ('\\underline{%s} & ' if model == OUR_MODEL else '%s & ') % model
                    cur_str += ' & '.join(['%s' % ('\\textbf{%s}' % t_records[model][metrics][task][str(threshold)]
                                                   if model == OUR_MODEL
                                                   else t_records[model][metrics][task][str(threshold)])
                                           for threshold in EVAL_METRICS_THRESHOLD_SET]) + \
                               ' \\\\\n'
                    if model == OUR_MODEL:
                        cur_str += '\\cline{2-%d}\n' % (2 + len(EVAL_METRICS_THRESHOLD_SET))
                    render_str += cur_str
            render_str += '\\hline\n'
        # Table Meta End
        render_str += '\\end{tabular}\n' + \
                      '\\end{table}'
        metrics_tables[metrics] = render_str
    return metrics_tables


if __name__ == '__main__':
    """ 
        Usage Example:
        python ResultTableRenderer.py -i ./dc_dataset/ -d "Washington D.C. (2017) 0101-0331"
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--in_dir', type=str, default=LOG_IN_DIR_DEFAULT, help='The directory for the input logs, default = {}'.format(LOG_IN_DIR_DEFAULT))
    parser.add_argument('-d', '--ds_tag', type=str, default=DS_TAG_DEFAULT, help='The tag name for the dataset, default = {}'.format(DS_TAG_DEFAULT))

    FLAGS, unparsed = parser.parse_known_args()

    total_records = summarizeRecInDir(FLAGS.in_dir)
    tables = renderTable(total_records, FLAGS.ds_tag)
    for metrics in tables:
        # print(metrics)
        print(tables[metrics])
        print()
