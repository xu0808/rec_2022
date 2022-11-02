# -*- coding: utf-8 -*-
# 召回服务
# 1、各召回源评分归一化
# 2、各召回源通过不同权重合并

import reader_2


def merge_recall(user_id, item_ids, topN):
    """ 多召回源整合 """
    # 所有召回源
    item_cf_i2i = reader_2.read_sim_file('1_item_cf_i2i')
    # user_cf_u2u = reader_2.read_sim_file('2_user_cf_u2u')
    mf_i2i = reader_2.read_sim_file('3_mf_i2i')
    mf_u2i = reader_2.read_sim_file('3_mf_u2i')
    fm_i2i = reader_2.read_sim_file('4_fm_i2i')
    fm_u2i = reader_2.read_sim_file('4_fm_u2i')

    # 所有召回源结果
    item_cf_i2i_recall = []
    mf_i2i_recall = []
    fm_i2i_recall = []
    for item_id in item_ids:
        item_cf_i2i_recall += item_cf_i2i[item_id]
        mf_i2i_recall += mf_i2i[item_id]
        fm_i2i_recall += fm_i2i[item_id]
    mf_u2i_recall = mf_u2i[user_id]
    fm_u2i_recall = fm_u2i[user_id]

    # 召回源权重配置
    recall_weight = []
    recall_weight.append((item_cf_i2i_recall, 1.0))
    recall_weight.append((mf_i2i_recall, 1.0))
    recall_weight.append((fm_i2i_recall, 1.0))
    recall_weight.append((mf_u2i_recall, 2.0))
    recall_weight.append((fm_u2i_recall, 3.0))

    # 最终召回排序分
    item_rec = {}
    for item_rank, weight in recall_weight:
        print('recall number = ', len(item_rank))
        # 当前召回源所有评分
        scores = [rec[1] for rec in item_rank]
        max_value = max(scores)
        min_value = min(scores)
        for item, score in item_rank:
            # 过滤当前召回内最小值
            if score == min_value:
                continue
            item_rec.setdefault(item, 0)
            # 每个召回源归一化后*权重累计结果
            item_rec[item] += weight * (score - min_value) / (max_value - min_value)
    result = sorted(item_rec.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)[:topN]
    print('recall result = ', result)
    return result


if __name__ == '__main__':
    merge_recall(user_id=96613, item_ids=[160417, 156624, 156447], topN=20)
