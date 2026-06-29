# 指标体系权重自优化算法
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import RRDA
from AII import AII
from FCE_AII_new import FCE_AII
import warnings
from model import Net,Net_GRU
from params import RESULT_VECTOR_MAPPING
import argparse
import util
import os
import draw_utils
import chardet


def read_csv_auto_encoding(file_path, **kwargs):
    """自动识别CSV编码，并在识别失败时尝试常见中英文编码。"""
    with open(file_path, 'rb') as file:
        raw_data = file.read(1024 * 1024)

    detected = chardet.detect(raw_data).get('encoding')
    candidates = [detected, 'utf-8-sig', 'utf-8', 'gb18030']
    tried = []
    last_error = None
    for encoding in candidates:
        if not encoding or encoding.lower() in tried:
            continue
        tried.append(encoding.lower())
        try:
            return pd.read_csv(file_path, encoding=encoding, **kwargs)
        except UnicodeDecodeError as exc:
            last_error = exc

    raise UnicodeError(
        f'无法识别CSV文件编码: {file_path}，已尝试: {", ".join(tried)}'
    ) from last_error

warnings.simplefilter("ignore", category=RuntimeWarning)

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

np.random.seed(2024)
torch.manual_seed(2024)


def get_init_weights(A_II, A_I, A_II_labels):
    # 根据一级指标权重集和二级指标权重集计算综合指标，得到init_weights (16, 1)
    for ai_k, ai_v in A_I.items():
        for aii_k, aii_v in A_II[ai_k].items():
            A_II[ai_k][aii_k] = ai_v * aii_v

    # 将A_II转换为权重矩阵,要与AII_labels对应
    A_II_weights = []
    for labels in A_II_labels:
        for ai_k, ai_v in A_II.items():
            if labels in A_II[ai_k].keys():
                A_II_weights.append(A_II[ai_k][labels])
                print(f"标签：{labels} 初始权重：{A_II[ai_k][labels]}")
                break
    # for aii_k, aii_v in A_II['学生本人情况'].items():
    #     A_II_weights.append(aii_v)
    # for aii_k, aii_v in A_II['学生家庭成员情况'].items():
    #     A_II_weights.append(aii_v)
    # for aii_k, aii_v in A_II['家庭经济情况'].items():
    #     A_II_weights.append(aii_v)
    A_II_weights = np.array(A_II_weights).reshape(-1, 1)
    # print("初始权重集：")
    # print(A_II_weights)

    return A_II_weights


def trans(w):
    # Step 1: Take the logarithm of the normalized weights
    lognw = torch.log(w)

    # Step 2: Subtract the mean of the log values to ensure numerical stability
    mean_log_weights = lognw.mean()
    rw = lognw - mean_log_weights

    return rw


# def get_normalized_weights(weights):
#     """ 归一化权重. 将（16, *）的权重归一化为（16, 1） """

#     mean_weights = weights.mean(dim=1, keepdim=True)
#     normalized_weights = torch.softmax(mean_weights, dim=0)
#     return normalized_weights


def get_labels_by_weights(data, weights, add_weight_noise=False):
    """ 根据权重生成标签 """
    if add_weight_noise:
        y = np.dot(data, weights) + np.random.normal(0, 0.001,
                                                     (data.shape[0], 1))
    else:
        y = np.dot(data, weights)
    y_class = np.zeros_like(y)

    # 按照y从大到小排序，分成4类，每类的数量比例为[0.4, 0.3, 0.2, 0.1]
    sorted_idx = np.argsort(y.flatten())[::-1]
    len_sorted_idx = len(sorted_idx)
    y_class[sorted_idx[:int(len_sorted_idx * 0.4)]] = 0
    y_class[sorted_idx[int(len_sorted_idx * 0.4):int(len_sorted_idx *
                                                     0.7)]] = 1
    y_class[sorted_idx[int(len_sorted_idx * 0.7):int(len_sorted_idx *
                                                     0.9)]] = 2
    y_class[sorted_idx[int(len_sorted_idx * 0.9):]] = 3

    # edges = [0.45, 0.5, 0.55]
    # y_class[y < edges[0]] = 0
    # y_class[(y >= edges[0]) & (y < edges[1])] = 1
    # y_class[(y >= edges[1]) & (y < edges[2])] = 2
    # y_class[y >= edges[2]] = 3
    return y_class.astype(int).flatten()


def load_data(
    sample_num=10000,
    real_data=None,
    device='cpu',
):
    """ 生成示例数据 """
    if real_data is not None:
        data = real_data[0].to_numpy().astype(np.float32)
        y_class = real_data[1].astype(int)
        weights = []
    else:
        data = np.random.rand(sample_num, 16).astype(np.float32)
        # 生成预期的权重
        weights = np.random.rand(16, 1)
        weights = weights / np.sum(weights)
        y_class = get_labels_by_weights(data, weights, add_weight_noise=True)

    # 将数据转换为PyTorch张量
    data = torch.tensor(data).to(device)
    y_class = torch.tensor(y_class, dtype=torch.long).squeeze().to(device)

    # 不再划分训练集和测试集,使用全部数据进行训练
    dataset = [data, y_class]

    return dataset, weights


def encode(data, df_zbtx, with_RRDA=True):
    """ 数据编码 """

    def trans_score(row, df):
        scores = {}
        for indicator_2, indicator_3, score in zip(df['indicator_2'],
                                                   df['indicator_3'],
                                                   df['normalized_score']):
            if (row[indicator_2] == indicator_3):
                scores[indicator_2] = score
        return scores

    all_scores = []
    indicator_2 = df_zbtx['indicator_2'].unique().tolist()
    df_data = read_csv_auto_encoding(data)
    for index, row in df_data.iterrows():
        scs_dict = trans_score(row, df_zbtx)
        for indicator in indicator_2:
            if indicator not in scs_dict.keys():
                scs_dict[indicator] = 0.0
        all_scores.append(scs_dict)

    if with_RRDA:
        # RRDA
        city_income_data = './data/全国各城市人均收入&最低收入.csv'

        # Load data
        df_city_income, df_std = RRDA.load_data(city_income_data=city_income_data,
                                                std_data=data)
        assert df_std.shape[0] == df_data.shape[0]
        df_std = RRDA.RRDA(df_city_income, df_std, alpha=0.279)
        gamas = df_std['家庭经济困难度指数'].to_numpy()

        # 将gamas放到all_scores的每一个字典中
        for i in range(len(all_scores)):
            all_scores[i]['家庭经济困难度指数'] = gamas[i]

    return all_scores


def weights_to_dfidx(weights, df_zbtx, A_II_labels, overlap=False):
    """ 将权重转换为指标体系 """
    df_zbtx['optimized_weights'] = df_zbtx['indicator_2'].apply(
        lambda x: weights[A_II_labels.index(x)] if x in A_II_labels else None)

    optimized_score_max = 'score_max' if overlap else 'optimized_score_max'
    optimized_score = 'score' if overlap else 'optimized_score'

    df_zbtx[optimized_score_max] = df_zbtx['optimized_weights'] * 100
    df_zbtx[optimized_score] = df_zbtx[optimized_score_max] * df_zbtx[
        'normalized_score']

    return df_zbtx


# def weights_to_A_I_A_II(weights):
#     """ 将权重转换为一级指标权重集和二级指标权重集 """
#     new_A_I = {
#         '学生本人情况': 0.0,
#         '学生家庭成员情况': 0.0,
#         '家庭经济情况': 0.0,
#     }

#     new_A_II = {
#         '学生本人情况': {},
#         '学生家庭成员情况': {},
#         '家庭经济情况': {},
#     }

#     for i, label in enumerate(A_II_labels):
#         if i < 3:
#             new_A_I['学生本人情况'] += weights[i]
#             new_A_II['学生本人情况'][label] = weights[i]
#         elif i < 11:
#             new_A_I['学生家庭成员情况'] += weights[i]
#             new_A_II['学生家庭成员情况'][label] = weights[i]
#         else:
#             new_A_I['家庭经济情况'] += weights[i]
#             new_A_II['家庭经济情况'][label] = weights[i]

#     for aii in new_A_II.values():
#         total = sum(aii.values())
#         for k, v in aii.items():
#             aii[k] = v / total

#     new_A_II['家庭经济情况']['家庭经济困难度指数等级'] = new_A_II['家庭经济情况'].pop('家庭经济困难度指数')

#     return new_A_I, new_A_II


def push_new_weights(new_weights, new_zbtx_path, old_zbtx_file, A_II_labels,
                     init_weights):
    """ 将新的权重保存到新的指标体系中 """
    df_zbtx = pd.read_csv(old_zbtx_file)
    df_zbtx = weights_to_dfidx(new_weights, df_zbtx, A_II_labels)
    df_zbtx.to_csv(os.path.join(new_zbtx_path, 'new_zbtx.csv'))

    # 总结权重信息到一个新的dataframe
    init_weights = np.round(init_weights, 4)
    init_weights = np.array(init_weights).reshape(-1, 1)
    # 保留4位小数
    new_weights = np.round(new_weights, 4)
    new_weights = new_weights.reshape(-1, 1)
    change = new_weights - init_weights
    change = np.round(change, 4)

    df_summary = pd.DataFrame({
        '指标名称':
        A_II_labels,
        '原始权重':
        init_weights.flatten(),
        '优化后权重':
        new_weights.flatten(),
        '权重变化趋势':
        np.where(
            new_weights.flatten() > init_weights.flatten(),
            '上升',
            '下降',
        ),
        '权重变化量':
        change.flatten(),
        '变化百分比 (%)':
        abs(np.round((change.flatten() / init_weights.flatten()) * 100, 2)),
    })

    # 保存到新的指标体系文件夹中
    df_summary.to_csv(os.path.join(new_zbtx_path, 'weight_summary.csv'),
                      index=False)

    return df_summary


def get_labels_by_AII(full_df_std, weights, zbtx_path, indices, A_II_labels,
                      A_I, A_II):
    """ 使用AII算法获取认定结果标签 """
    # 将full_df_std中的数据copy到df_std中
    df_std = full_df_std.copy()
    df_zbtx = pd.read_csv(zbtx_path)
    df_idx = weights_to_dfidx(weights, df_zbtx, A_II_labels, overlap=True)
    # print(df_idx)

    # RRDA_score = A_II[RRDA_AI_indicater_1]['家庭经济困难度指数'] * A_I[
    # RRDA_AI_indicater_1] * 100

    df_std = AII(
        df_std,
        save=False,
        df_idx=df_idx,
        #  RRDA_score=RRDA_score,
        indices=indices, patch=True)

    # 将df_std中的结果（实际认定结果列）转换为标签
    result = df_std['算法认定结果']
    # 释放内存
    del df_std
    labels = result.map(RESULT_VECTOR_MAPPING).to_numpy()
    return labels


def get_labels_by_FCEAII(full_df_std, weights, indices, zbtx_file):
    """ 根据FCEAII计算标签 """
    df_std = full_df_std.copy()
    # new_A_I, new_A_II = weights_to_A_I_A_II(weights)
    full_df_std = FCE_AII(
        df_std,
        zbtx_file=zbtx_file,
        save=False,
        indices=indices,
        with_RRDA=False,
    )

    # acc = util.calc_acc(full_df_std, '算法认定结果', '实际认定结果')
    # print(f'一致性：{acc * 100 :.2f}%')
    # print("FCEAII 计算完成")

    result = full_df_std['算法认定结果']
    # 释放内存
    del df_std
    labels = result.map(RESULT_VECTOR_MAPPING).to_numpy()
    return labels


def soft_macro_recall_loss(model_outputs, true_labels, eps=1e-8):
    """可反向传播的宏平均召回率损失。"""
    probabilities = torch.softmax(model_outputs, dim=1)
    one_hot_labels = torch.nn.functional.one_hot(
        true_labels, num_classes=model_outputs.shape[1]
    ).to(dtype=probabilities.dtype)
    true_positives = (probabilities * one_hot_labels).sum(dim=0)
    actual_positives = one_hot_labels.sum(dim=0)
    present_classes = actual_positives > 0
    recalls = true_positives[present_classes] / (
        actual_positives[present_classes] + eps
    )
    return 1.0 - recalls.mean()


def macro_recall_score(predicted, true_labels, num_classes):
    """计算实际预测结果的宏平均召回率。"""
    recalls = []
    for class_id in range(num_classes):
        class_mask = true_labels == class_id
        actual_count = class_mask.sum().item()
        if actual_count > 0:
            true_positive = ((predicted == class_id) & class_mask).sum().item()
            recalls.append(true_positive / actual_count)
    return float(np.mean(recalls)) if recalls else 0.0


def combined_loss2(criterion, model_outputs, true_labels, indices, full_df_std,
                   weights, old_zbtx_file, alpha, beta, recall_weight,
                   A_II_labels, A_I, A_II):
    """ 计算组合损失函数 """
    # 计算模型输出的交叉熵损失
    ce_loss = criterion(
        model_outputs,
        true_labels,
    )
    recall_loss = soft_macro_recall_loss(model_outputs, true_labels)
    supervised_loss = (
        (1 - recall_weight) * ce_loss
        + recall_weight * recall_loss
    )

    # 计算直接用权重加权求和得到的标签
    # weights = weights.detach().cpu().numpy()  # 确保 weights 不计算梯度
    # print(weights)
    AII_loss, FCEAII_loss = 0, 0
    if alpha > 0:
        AII_labels = get_labels_by_AII(full_df_std, weights, old_zbtx_file,
                                       indices, A_II_labels, A_I, A_II)
        AII_labels = torch.tensor(AII_labels,
                                  dtype=torch.long,
                                  device=true_labels.device)
        AII_loss = criterion(model_outputs, AII_labels)
    if beta > 0:
        # print(full_df_std.shape)
        FCEAII_labels = get_labels_by_FCEAII(full_df_std, weights, None,
                                             old_zbtx_file)
        FCEAII_labels = FCEAII_labels[indices]

        FCEAII_labels = torch.tensor(FCEAII_labels,
                                     dtype=torch.long,
                                     device=true_labels.device)
        FCEAII_loss = criterion(model_outputs, FCEAII_labels)

    # 计算总损失
    total_loss = (
        (1 - alpha - beta) * supervised_loss
        + alpha * AII_loss
        + beta * FCEAII_loss
    )
    return total_loss


def train(model,
          optimizer,
          scheduler,
          dataset,
          num_epochs=200,
          batch_size=32,
          criterion=nn.CrossEntropyLoss(),
          full_df_std=None,
          old_zbtx_file=None,
          alpha=0.33,
          beta=0.33,
          recall_weight=0.3,
          device='cpu',
          callback=None,
          A_II_labels=None,
          A_I=None,
          A_II=None,
          init_weights=None,
          work_dir=None):
    """ 训练模型 """

    losses = []
    train_accs = []
    aii_test_accs = []
    fceaii_test_accs = []
    normalized_weights_list = []

    data_all, y_all = dataset

    model.train()
    print('开始训练...')

    for epoch in range(num_epochs):

        # 记录权重变化
        normalized_weights = model.get_weights()
        normalized_weights_list.append(normalized_weights)

        # 随机打乱数据
        permutation = torch.randperm(data_all.size()[0])
        loss = 0
        for i in range(0, data_all.size()[0], batch_size):

            indices = permutation[i:i + batch_size]
            batch_x, batch_y = data_all[indices], y_all[indices]

            optimizer.zero_grad()  # 梯度清零
            outputs = model(batch_x)  # 前向传播

            # 计算损失
            batch_loss = combined_loss2(
                criterion,
                outputs,
                batch_y,
                indices.detach().cpu().numpy(),
                full_df_std,
                model.get_weights(),
                old_zbtx_file,
                alpha=alpha,
                beta=beta,
                recall_weight=recall_weight,
                A_II_labels=A_II_labels,
                A_I=A_I,
                A_II=A_II,
            )

            batch_loss.backward()  # 反向传播
            loss += batch_loss.item()
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
        losses.append(loss)

        with torch.no_grad():
            model.eval()
            # 计算模型在全部数据上的一致率
            outputs = model(data_all)
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == y_all).sum().item()
            train_acc = correct / y_all.size(0)
            train_recall = macro_recall_score(
                predicted, y_all, outputs.shape[1]
            )
            train_accs.append(train_acc)

            # 计算模型在全部数据上执行AII、FCEAII的一致率
            AII_test_acc_weight = -1
            if alpha > 0:
                AII_predicted = get_labels_by_AII(full_df_std,
                                                  model.get_weights(),
                                                  old_zbtx_file, None,
                                                  A_II_labels, A_I, A_II)

                AII_test_acc_weight = (AII_predicted == y_all.cpu().numpy()
                                       ).sum() / y_all.size(0)

            FCE_AII_test_acc_weight = -1
            if beta > 0:
                FCE_AII_predicted = get_labels_by_FCEAII(
                    full_df_std, model.get_weights(), None, old_zbtx_file)
                FCE_AII_test_acc_weight = (
                    FCE_AII_predicted
                    == y_all.cpu().numpy()).sum() / y_all.size(0)

            aii_test_accs.append(AII_test_acc_weight)
            fceaii_test_accs.append(FCE_AII_test_acc_weight)

            model.train()

        # if (epoch + 1) % 10 == 0:
        print(
            f'Epoch [{epoch + 1}/{num_epochs}]\tLoss: {loss:.4f}\tacc: {train_acc:.4f}\tRecall: {train_recall:.4f}\tAII: {AII_test_acc_weight:.4f},\tFCEAII: {FCE_AII_test_acc_weight:.4f}'
        )
        # 绘制损失曲线
        loss_img = draw_utils.draw_plot(losses,
                             '训练损失变化',
                             '训练轮数',
                             '损失值')

        acc_img = draw_utils.draw_plot(aii_test_accs,
                             'AII算法一致率变化',
                             '训练轮数',
                             '一致率')

        # 绘制所有权重变化叠加图（替换准确率图）
        acc_aii_fce_img, color_mapping = draw_utils.draw_all_weights_overlay(
                              normalized_weights_list,
                              A_II_labels)

        # 绘制权重变化
        weight_img = draw_utils.draw_weights(normalized_weights_list,
                                init_weights,
                                A_II_labels)

        if callback:
            # 准备回调数据
            weights_dict = {
                label: weight
                for label, weight in zip(A_II_labels, normalized_weights)
            }
            continue_training = callback(epoch + 1, loss, train_acc, train_acc,
                                         train_recall,
                                         AII_test_acc_weight,
                                         FCE_AII_test_acc_weight, weights_dict,
                                         loss_img, acc_img, acc_aii_fce_img, weight_img, color_mapping)
            if not continue_training:
                break  # 如果回调返回False，停止训练

    # 在全部数据上评估模型最终结果
    model.eval()
    with torch.no_grad():
        outputs = model(data_all)
        _, predicted = torch.max(outputs, 1)
        final_acc = (predicted == y_all).sum().item() / y_all.size(0)

        AII_test_acc_weight = -1
        if alpha > 0:
            AII_predicted = get_labels_by_AII(full_df_std, model.get_weights(),
                                              old_zbtx_file, None, A_II_labels,
                                              A_I, A_II)
            AII_test_acc_weight = (
                np.array(AII_predicted)
                == y_all.cpu().numpy()).sum() / y_all.size(0)

        FCE_AII_test_acc_weight = -1
        if beta > 0:
            FCE_AII_predicted = get_labels_by_FCEAII(full_df_std,
                                                     model.get_weights(), None,
                                                     old_zbtx_file)
            FCE_AII_test_acc_weight = (
                np.array(FCE_AII_predicted)
                == y_all.cpu().numpy()).sum() / y_all.size(0)
        aii_test_accs.append(AII_test_acc_weight)
        fceaii_test_accs.append(FCE_AII_test_acc_weight)

    return losses, train_accs, final_acc, aii_test_accs, fceaii_test_accs, normalized_weights_list


def self_optimization(full_df_std,
                      real_data,
                      old_zbtx_file,
                      new_zbtx_path,
                      expected_weights,
                      lr,
                      epochs,
                      batch_size,
                      alpha,
                      beta,
                      A_I,
                      A_II,
                      A_II_labels,
                      device='cuda',
                      recall_weight=0.3,
                      callback=None,
                      work_dir=None):
    """ 自优化算法主函数 """

    if not 0 <= recall_weight <= 1:
        raise ValueError('召回率损失权重 recall_weight 必须在 0 和 1 之间')
    if alpha < 0 or beta < 0 or alpha + beta > 1:
        raise ValueError('alpha、beta 必须非负，并且 alpha + beta 不能超过 1')

    # 超参数
    model = Net()
    # model = Net()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=lr,
    )
    # scheduler = None
    # 初始化权重
    init_weights = get_init_weights(A_II, A_I, A_II_labels)
    A_II_weights = torch.tensor(init_weights, dtype=torch.float32).to(device)
    model.fc1.weight.data = trans(A_II_weights)

    # 加载数据
    dataset = load_data(real_data=real_data, device=device)[0]

    training_outputs = train(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        dataset=dataset,
        num_epochs=epochs,
        batch_size=batch_size,
        criterion=criterion,
        full_df_std=full_df_std,
        old_zbtx_file=old_zbtx_file,
        alpha=alpha,
        beta=beta,
        recall_weight=recall_weight,
        device=device,
        callback=callback,
        A_II_labels=A_II_labels,
        A_I=A_I,
        A_II=A_II,
        init_weights=init_weights,
        work_dir=work_dir,
    )

    losses, train_accs, final_acc, aii_test_accs, fceaii_test_accs, normalized_weights_list = training_outputs

    print(f'Final Accuracy: {final_acc:.4f},\
        Accuracy of AII: {aii_test_accs[-1]:.4f}, FCEAII: {fceaii_test_accs[-1]:.4f}'
          )

    # 保存权重变化
    import pandas as pd
    df = pd.DataFrame(normalized_weights_list)
    weight_change_path = os.path.join(work_dir, 'weight_change.csv')
    df.to_csv(weight_change_path, index=False)

    if expected_weights is not None:
        print('期待权重:')
        print(expected_weights)

    final_weights = normalized_weights_list[-1]
    print('最终权重:')
    print(final_weights)

    # 将最终权重置入指标体系中
    weight_summary_df = None
    if new_zbtx_path is not None:
        weight_summary_df = push_new_weights(
            normalized_weights_list[-1],
            new_zbtx_path,
            old_zbtx_file,
            A_II_labels,
            init_weights,
        )

    # 绘制损失曲线
    draw_utils.draw_plot(losses, '训练损失变化', '训练轮数', '损失值', work_dir=work_dir)

    # 绘制AII算法一致率变化
    draw_utils.draw_plot(aii_test_accs,
                         'AII算法一致率变化',
                         '训练轮数',
                         '一致率',
                         file='acc_AII_FCE.png',
                         work_dir=work_dir)

    # 绘制模型一致率曲线
    draw_utils.draw_plot(train_accs,
                         '模型一致率变化',
                         '训练轮数',
                         '一致率',
                         file='acc_train.png',
                         work_dir=work_dir)

    # 绘制权重变化
    draw_utils.draw_weights(normalized_weights_list,
                            init_weights,
                            A_II_labels,
                            work_dir=work_dir)

    return weight_summary_df


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--old_zbtx_file',
        type=str,
        default='tmp/整合版量化指标体系.csv',
        # default='data/指标体系_v1.csv',
        help='原始指标体系文件路径')
    parser.add_argument('--new_zbtx_path',
                        type=str,
                        default='tmp',
                        help='优化后的指标体系文件保存路径')
    parser.add_argument(
        '--dataset',
        type=str,
        default='tmp/新模拟数据.csv',
        # default='data/base/raw_500_modified.csv',
        help='含认定结果的认定数据文件路径')
    parser.add_argument('--lr', type=float, default=1e-2, help='学习率，默认值为0.01')
    parser.add_argument('--epochs', type=int, default=10, help='训练轮数，默认值为10')
    parser.add_argument('--batch_size',
                        type=int,
                        default=128,
                        help='训练批次大小，默认值为128')
    parser.add_argument('--alpha',
                        type=float,
                        default=0.3,
                        help='AII自优化权重，默认值为0.4')
    parser.add_argument('--beta',
                        type=float,
                        default=0.,
                        help='FCE自优化权重，默认值为0')
    parser.add_argument('--recall_weight',
                        type=float,
                        default=0.3,
                        help='Soft Macro-Recall损失权重，范围[0, 1]，默认值为0.3')
    parser.add_argument('--device', type=str, default='cpu', help='训练设备')
    args = parser.parse_args()

    dataset = args.dataset

    real_result = read_csv_auto_encoding(dataset)['实际认定结果']

    real_labels = real_result.map(RESULT_VECTOR_MAPPING).to_numpy()

    # 获取二级指标列表
    df_zbtx = read_csv_auto_encoding(args.old_zbtx_file)
    A_I, A_II = util.get_AI_AII(df_zbtx)
    A_II_labels = []
    for ai in A_II.keys():
        for aii in A_II[ai].keys():
            A_II_labels.append(aii)

    encode_data = encode(dataset,
                         df_zbtx=read_csv_auto_encoding(args.old_zbtx_file),
                         with_RRDA=False)
    # 将encode_data中的每个字典按照 A_II_labels 的顺序进行排序
    for i in range(len(encode_data)):
        encode_data[i] = {
            k: v
            for k, v in sorted(encode_data[i].items(),
                               key=lambda item: A_II_labels.index(item[0]))
        }
    # 将encode_data转换为DataFrame
    df = pd.DataFrame(encode_data)
    # 保存到csv文件
    # df.to_csv('data/tmp/encode_data.csv', index=False)

    # 执行RRDA算法后的数据集
    # full_df_std = RRDA.run_RRDA(dataset)
    full_df_std = read_csv_auto_encoding(dataset)

    # 编码后的数据集
    real_data = [df, real_labels]

    self_optimization(
        full_df_std,
        real_data,
        args.old_zbtx_file,
        args.new_zbtx_path,
        None,
        args.lr,
        args.epochs,
        args.batch_size,
        args.alpha,
        args.beta,
        A_I,
        A_II,
        A_II_labels,
        args.device,
        recall_weight=args.recall_weight,
        work_dir=args.new_zbtx_path,
    )

    # example: python selfOptimization.py
    # --new_zbtx_file data/tmp/new_zbtx.csv --lr 0.01 --epochs 100 --batch_size 128 --alpha 0.3 --beta 0.
#  TODO 尝试优化三级指标
