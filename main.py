import os
import warnings

import dill
import librosa
import librosa.feature
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from xgboost import XGBClassifier

warnings.simplefilter("ignore")  # 忽略警告
warnings.filterwarnings("ignore")  # 忽略警告
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]  # 用来正常显示中文标签
plt.rcParams["axes.unicode_minus"] = False  # 用来正常显示负号

IDX2LABELS = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}  # 定义一个字典，用数字表示RAVDESS数据集中的情绪
LABELS = list(IDX2LABELS.values())  # 情绪标签


def save_pkl(filepath, data):
    # 保存模型
    with open(filepath, "wb") as fw:
        dill.dump(data, fw)
    print(f"[{filepath}] data saving...")


def load_pkl(filepath):
    # 加载模型
    with open(filepath, "rb") as fr:
        data = dill.load(fr, encoding="utf-8")
    print(f"[{filepath}] data loading...")
    return data


def save_txt(filepath, data):
    # 保存文本
    with open(filepath, "w", encoding="utf-8") as fw:
        fw.write(data)
    print(f"{filepath} saving...")


def standard_scaler(values, scaler_path, mode="train"):
    # 数据标准化
    if mode == "train":
        scaler = StandardScaler()  # 定义标准化模型
        scaler.fit(values)  # 训练模型
        save_pkl(scaler_path, scaler)  # 保存模型
    else:
        scaler = load_pkl(scaler_path)  # 加载模型
    return scaler.transform(values)  # 对数据进行转换


def save_evaluate(y_test, y_test_pred, outputs_path):
    # 保存性能指标和混淆矩阵
    test_report = classification_report(
        y_test,
        y_test_pred,
        labels=list(range(len(LABELS))),
        target_names=LABELS,
        digits=4,
    )  # 计算测试集性能指标 包括precision/recall/f1-score/accuracy
    test_matrix = confusion_matrix(y_test, y_test_pred)  # 计算测试集混淆矩阵

    results = "test classification report:\n" + test_report + "\nconfusion matrix\n" + str(test_matrix)  # 拼接性能指标和混淆矩阵
    save_txt(outputs_path, results)  # 保存性能指标和混淆矩阵
    print(results)  # 打印性能指标和混淆矩阵


def plot_confusion_matrix(y_test, y_test_pred, output_path):
    # 画混淆矩阵
    matrix = confusion_matrix(y_test, y_test_pred)  # 计算混淆矩阵
    matrix = matrix.astype("float") / matrix.sum(axis=1)[:, np.newaxis]  # 归一化

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6.4, 4.8), dpi=100)  # 定义画布
    sns.heatmap(matrix, annot=True, fmt=".2f", linewidths=0.5, square=True, cmap="Blues", ax=ax)  # 画热力图
    ax.set_title("混淆矩阵可视化")  # 标题
    ax.set_xlabel("真实标签")  # x轴标签
    ax.set_ylabel("预测标签")  # y轴标签
    ax.set_xticks([x + 0.5 for x in range(len(LABELS))], LABELS, rotation=0)  # x轴刻度
    ax.set_yticks([x + 0.5 for x in range(len(LABELS))], LABELS, rotation=0)  # y轴刻度
    plt.savefig(output_path)  # 保存图像
    # plt.show()  # 显示图像
    plt.close()  # 关闭图像


def plot_roc(y_test, y_test_pred_score, output_path):
    # 画ROC曲线
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6.4, 4.8), dpi=100)  # 定义画布
    for label_idx, label_name in enumerate(LABELS):  # 遍历所有标签
        false_positive_rate, true_positive_rate, _ = roc_curve(
            [1 if y == label_idx else 0 for y in y_test],
            y_test_pred_score[:, label_idx],
        )  # 计算ROC数值
        roc_auc = auc(false_positive_rate, true_positive_rate)  # 计算AUC
        ax.plot(false_positive_rate, true_positive_rate, label=f"{label_name}, AUC = {roc_auc:0.4f}")  # 画折线图
    ax.plot([0, 1], [0, 1], "r--")  # 画对角线
    ax.set_xlabel("False Positive Rate")  # x轴标签
    ax.set_ylabel("True Positive Rate")  # y轴标签
    ax.set_title("ROC curve based on RF model")  # 标题
    plt.legend(loc="lower right")  # 显示标签
    plt.savefig(output_path)  # 保存图像
    # plt.show()  # 显示图像
    plt.close()  # 关闭图像


def build_datasets(voice_root_dir, data_path, outputs_dir):
    # 根据音频数据提取特征并构建数据集
    data = []  # 数据集
    for voice_dir_idx, voice_dir in enumerate(os.listdir(voice_root_dir)):  # 遍历音频数据根目录下的子目录
        for voice_path_idx, voice_path in enumerate(os.listdir(os.path.join(voice_root_dir, voice_dir))):  # 遍历子目录下的音频文件
            print(f"正在处理第{voice_dir_idx + 1}个子目录的第{voice_path_idx + 1}个音频文件...")

            label = IDX2LABELS[voice_path.split(".")[0].split("-")[2]]  # 情绪标签
            gender = 0 if int(voice_path.split(".")[0].split("-")[6]) & 1 == 0 else 1  # 性别  0: female  1: male

            # 读取音频文件
            voice, sr = librosa.load(os.path.join(voice_root_dir, voice_dir, voice_path), sr=None)  # sr=None 保持原始采样率

            # log-mel spectrogram 特征提取
            mel_spect = librosa.feature.melspectrogram(y=voice, sr=sr, n_fft=2048, hop_length=512, n_mels=128)  # 梅尔频谱
            log_mel_spect = librosa.power_to_db(mel_spect)  # log转换为db
            log_mel_features = np.mean(log_mel_spect, axis=1)  # 沿着频谱轴计算均值

            # mfcc 特征提取
            mfcc = librosa.feature.mfcc(y=voice, sr=sr, n_mfcc=40)  # 梅尔频率倒谱系数
            mfcc_features = np.mean(mfcc, axis=1)  # 沿着频谱轴计算均值

            # 组合特征
            sample = {"label": label, "gender": gender}
            sample.update({f"log_mel_{idx}": value for idx, value in enumerate(log_mel_features)})
            sample.update({f"mfcc_{idx}": value for idx, value in enumerate(mfcc_features)})
            data.append(sample)

            if voice_dir_idx == 0 and voice_path_idx == 0:
                librosa.display.waveshow(voice, sr=sr)  # 绘制波形图
                # plt.show()  # 显示图像
                plt.savefig(os.path.join(outputs_dir, "voice.png"))  # 保存图像
                plt.close()  # 关闭图像

                librosa.display.specshow(log_mel_spect, sr=sr, x_axis="time", y_axis="mel")  # 绘制梅尔频谱
                # plt.show()  # 显示图像
                plt.savefig(os.path.join(outputs_dir, "voice_log_mel.png"))  # 保存图像
                plt.close()  # 关闭图像

                librosa.display.specshow(mfcc, sr=sr, x_axis="time")  # 绘制梅尔频率倒谱系数
                # plt.show()  # 显示图像
                plt.savefig(os.path.join(outputs_dir, "voice_mfcc.png"))  # 保存图像
                plt.close()  # 关闭图像

    data = pd.DataFrame(data)  # 转换为DataFrame
    data.to_csv(data_path, index=False)  # 保存数据集


if __name__ == "__main__":
    voice_root_dir = rf"voice-data"  # 音频数据根目录
    data_path = rf"data.csv"  # 数据集路径
    outputs_dir = rf"outputs"  # 输出目录
    os.makedirs(outputs_dir, exist_ok=True)  # 新建输出目录

    if not os.path.exists(data_path):  # 如果数据集不存在
        build_datasets(voice_root_dir, data_path, outputs_dir)  # 构建数据集

    data = pd.read_csv(data_path)  # 读取数据集
    X = data.drop(columns="label").values  # 特征
    y = data["label"].apply(lambda x: LABELS.index(x)).values  # 标签

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # 按照8:2划分训练集和测试集
    X_train = standard_scaler(X_train, os.path.join(outputs_dir, "continuous_scaler.pkl"), mode="train")  # 对特征进行标准化 训练集
    X_test = standard_scaler(X_test, os.path.join(outputs_dir, "continuous_scaler.pkl"), mode="test")  # 对特征进行标准化 测试集

    model = XGBClassifier(n_jobs=-1, random_state=42)  # 定义模型
    model.fit(X_train, y_train)  # 训练模型
    save_pkl(os.path.join(outputs_dir, "model.pkl"), model)  # 保存模型

    y_test_pred = model.predict(X_test)  # 测试集预测
    save_evaluate(y_test, y_test_pred, os.path.join(outputs_dir, "evaluate.txt"))  # 保存评估指标
    plot_confusion_matrix(y_test, y_test_pred, os.path.join(outputs_dir, "confusion_matrix.png"))  # 绘制混淆矩阵

    y_test_pred_score = model.predict_proba(X_test)  # 计算测试集的得分
    plot_roc(y_test, y_test_pred_score, os.path.join(outputs_dir, "roc.png"))  # 绘制ROC曲线
