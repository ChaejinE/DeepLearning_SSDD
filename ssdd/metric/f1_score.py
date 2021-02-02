import logging
import numpy as np


def calculate_precision(tp_num, fp_num):
    """
    precision 을 계산해주는 함수입니다.
    :param tp_num: true_positive 갯수 >= 0
    :param fp_num: false_positive 갯수 >= 0
    :return: precision
    """
    try:
        if tp_num >= 0 and fp_num >= 0:
            precision = tp_num / (tp_num + fp_num)
            return precision
        else:
            logging.info('[INPUT SIGN CHECK] '
                         '\n tp_num : {}, fp_num : {}'.format(tp_num, fp_num))
            return None

    except TypeError:
        logging.error('[TypeError] [INPUT TYPE CHECK] '
                      '\n input type 은 정수형이거나 실수형이어 햡니다.-> tp_num : {}, fp_num : {}'.
                      format(type(tp_num), type(fp_num)))
        return None

    except ZeroDivisionError:
        logging.error('[ZeroDivisionError] '
                      '\n 분모를 계산할 수 없습니다. -> tp_num : {}, fp_num :{}'.format(tp_num, fp_num))
        return 0


def calculate_recall(tp_num, fn_num):
    """
    recall 값을 계산해주는 함수입니다.
    :param tp_num: true_positive 갯수 >= 0
    :param fn_num: false_negative 갯수 >= 0
    :return: recall Type : float
    """
    try:
        if tp_num >= 0 and fn_num >= 0:
            recall = tp_num / (tp_num + fn_num)
            return recall
        else:
            logging.info('[INPUT SIGN CHECK] '
                         '\n tp_num : {}, fn_num : {}'.format(tp_num, fn_num))
            return None

    except TypeError:
        logging.error('[TypeError] [INPUT TYPE CHECK] '
                      '\n input type 은 정수형이거나 실수형이어 햡니다.-> tp_num : {}, fn_num : {}'.
                      format(type(tp_num), type(fn_num)))
        return None

    except ZeroDivisionError:
        logging.error('[ZeroDivisionError] '
                      '\n 분모를 계산할 수 없습니다. -> tp_num : {}, fn_num :{}'.format(tp_num, fn_num))

        return 0


def calculate_f1_score(precision, recall):
    """
    f1 score 를 게산해주는 함수입니다.
    :param precision: precision value 정수 또는 실수
    :param recall: recall value 정수 또는 실수
    :return: f1_score
    """
    try:
        if precision >= 0 and recall >= 0:
            f1_score = 2 * (precision * recall) / (precision + recall)
            return f1_score
        else:
            logging.info('[INPUT SIGN CHECK] '
                         '\n precision recall 은 0 이상이어야합니다. -> precision : {}, recall : {}'.format(precision, recall))
            return None

    except TypeError:
        logging.error('[TypeError] [INPUT TYPE CHECK]'
                      '\n precision recall 은 정수 또는 실수 여야합니다. -> precision : {}, recall : {}'.
                      format(type(precision), type(recall)))

    except ZeroDivisionError:
        logging.error('[ZeroDivisionError] '
                      '\n precision과 recall의 합이 0입니다. 계산을 진행할 수 없습니다. -> precision : {}, recall : {}'.
                      format(precision, recall))

        return 0


def count_f1(label, pred):
    """
    :param label: tensor of shape (batch size,)
    :param pred: tensor of shape (batch size, nrof class)
    :param tp: int
    :param fp: int
    :param fn: int
    :return:
    """
    if len(pred) != len(label):

        raise Exception('pred 와 label 의 길이가 다릅니다. 인자를 확인하여 주세요.')

    if len(np.shape(label)) != 1 or len(np.shape(pred)) != 2:

        raise Exception('shape 을 확인해주세요.')

    if np.shape(pred)[-1] != max(label) + 1:

        raise Exception('pred 의 class 수는 label 의 class 수와 동일해야 합니다.')

    tp, fp, fn = 0, 0, 0
    pred = np.argmax(pred, axis=-1)
    for _l, _p in zip(label, pred):
        if _l not in [0, 1]:

            raise ValueError('label, pred 의 원소는 0 또는 1 이어야 합니다.')

        if _l == _p:
            if _p:
                tp += 1
        else:
            if _p:
                fp += 1
            else:
                fn += 1

    return tp, fp, fn


def update_f1(tp, fp, fn, label, pred):
    _tp, _fp, _fn = count_f1(label, pred)
    tp = tp+_tp
    fp = fp+_fp
    fn = fn+_fn

    return tp, fp, fn
