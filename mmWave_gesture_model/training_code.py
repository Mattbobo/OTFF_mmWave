import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Dense, Conv2D, MaxPooling2D, Flatten, LSTM, Dropout, TimeDistributed, BatchNormalization, Reshape, Masking
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import StratifiedKFold
import pickle
from tensorflow.keras.regularizers import l2
from sklearn.utils import class_weight

# 检查是否有可用的 GPU
gpus = tf.config.list_physical_devices('GPU')

if gpus:
    try:
        # 设置只使用第一个 GPU
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        print("Using GPU: ", gpus[0])
    except RuntimeError as e:
        print("Error setting GPU: ", e)
else:
    print("No GPU found, using CPU.")

# 读取单个.h5文件数据，并按照label切割
def load_single_file(h5_path, num_frames):
    with h5py.File(h5_path, 'r') as hf:
        X = np.array(hf['DS1'])
        y = np.array(hf['LABEL']).flatten()
        gesture_names = list(hf['DATA_CONFIG'].attrs['Gesture_name'])
        gesture_names.append('no_gesture')  # 添加 no_gesture 标签
        gesture_map = {i: name for i, name in enumerate(gesture_names)}  # Start from 0

    valid_slices = []
    start = None
    for i, label in enumerate(y):
        if label != 0:
            if start is None:
                start = i
        elif start is not None:
            valid_slices.append((start, i))
            start = None
    if start is not None:
        valid_slices.append((start, len(y)))

    X_segments = []
    y_segments = []
    gesture_label = 0  # Start labeling gestures from 0
    for start, end in valid_slices:
        segment_length = end - start
        if segment_length < num_frames:
            padding_length = num_frames - segment_length
            # 填充值0
            padding = np.zeros((X.shape[0], X.shape[1], X.shape[2], padding_length))
            X_segment = np.concatenate([X[:, :, :, start:end], padding], axis=3)
        else:
            X_segment = X[:, :, :, start:start + num_frames]

        X_segments.append(X_segment)
        y_segments.append(gesture_label)  # Use the current label
        gesture_label += 1  # Increment for the next gesture

    X_segments = np.stack(X_segments) if X_segments else np.empty((0, X.shape[1], X.shape[2], num_frames))
    y_segments = np.array(y_segments)

    print(f'Loaded {h5_path}: {len(y_segments)} samples, labels: {np.unique(y_segments)}')

    return X_segments, y_segments, gesture_map

def load_data(h5_dir, train_ratio=0.8, val_ratio=0.1, num_frames=35):
    file_names = [f for f in os.listdir(h5_dir) if f.endswith('.h5')]
    np.random.shuffle(file_names)
    num_files = len(file_names)
    num_train = int(num_files * train_ratio)
    num_val = int(num_files * val_ratio)
    num_test = num_files - num_train - num_val

    train_files = file_names[:num_train]
    val_files = file_names[num_train:num_train + num_val]
    test_files = file_names[num_train + num_val:]

    def load_files(file_list):
        X_list = []
        y_list = []
        gesture_map = None
        for file_name in file_list:
            file_path = os.path.join(h5_dir, file_name)
            X, y, gesture_map = load_single_file(file_path, num_frames)
            X_list.append(X)
            y_list.append(y)
        X = np.concatenate(X_list, axis=0)
        y = np.concatenate(y_list, axis=0)
        return X, y, gesture_map

    X_train, y_train, gesture_map = load_files(train_files)
    X_val, y_val, _ = load_files(val_files)
    X_test, y_test, _ = load_files(test_files)

    print(X_train.shape) # number_sample, depth, height, width, frame

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), gesture_map

# 数据预处理
def preprocess_data(X, y, num_cate, mean=None, std=None, sav=False):
    if mean is None or std is None:
        mean = np.mean(X, axis=(0, 1, 2, 3))
        std = np.std(X, axis=(0, 1, 2, 3))

        # 保存均值和标准差，仅在处理训练数据时
        if sav:
            with open('normalization_params.pkl', 'wb') as f:
                pickle.dump({'mean': mean, 'std': std}, f)

    X_normalized = (X - mean) / std
    y_one_hot = tf.keras.utils.to_categorical(y, num_classes=num_cate)

    return X_normalized, y_one_hot, mean, std


def create_model(input_shape, num_classes, learning_rate=0.0001):
    model = Sequential()

    # 第一层卷积层和池化层
    model.add(Conv3D(16, kernel_size=(3, 3, 3), padding='same', activation='relu', input_shape=input_shape))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))

    # 第二层卷积层和池化层
    model.add(Conv3D(32, kernel_size=(3, 3, 3), padding='same', activation='relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))

    # 第三层卷积层和池化层
    model.add(Conv3D(64, kernel_size=(3, 3, 3), padding='same', activation='relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))

    # 展平数据以准备输入LSTM
    model.add(TimeDistributed(Flatten()))

    # 第一层 LSTM 层，用于捕捉时间序列特征
    model.add(LSTM(64, return_sequences=False))

    # 全连接层
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))  # 添加 Dropout 层以防止过拟合

    # 输出层
    model.add(Dense(num_classes, activation='softmax'))

    # 使用 Adam 优化器
    optimizer = Adam(learning_rate=learning_rate)

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model

# 训练模型
def train_model(model, X_train, y_train, X_val, y_val, epochs=15, batch_size=128):
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    # 使用ReduceLROnPlateau调度器来减少学习率
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1)

    # 假设你知道所有类别的数量
    num_classes = 6

    # 初始化 class_weight_dict，所有类别的权重初始为 1
    class_weight_dict = {i: 1.0 for i in range(num_classes)}

    # 为特定的类别标签（比如 target_class_label）设置较高的权重
    # 假设你想要给类别 1 更高的权重
    class_weight_dict[1] = 1.4  # 0.5~10
    # class_weight_dict[0] = 3.0  # 0.5~10
    # class_weight_dict[3] = 2  # 0.5~10
    # class_weight_dict[4] = 0.5  # 0.5~10
    

    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                        validation_data=(X_val, y_val), callbacks=[early_stopping, lr_scheduler], class_weight=class_weight_dict)

    return history

# 评估模型
def evaluate_model(model, X_test, y_test, gesture_map):
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Test Loss: {loss}')
    print(f'Test Accuracy: {accuracy}')

    y_pred = model.predict(X_test)
    y_test_labels = np.argmax(y_test, axis=1)
    y_pred_labels = np.argmax(y_pred, axis=1)
    existing_labels = np.unique(y_test_labels)

    report = classification_report(y_test_labels, y_pred_labels,
                                   labels=existing_labels,
                                   target_names=[gesture_map[i] for i in existing_labels])
    print(report)

# 绘制训练过程的图表
def plot_training_history(history):
    epochs = range(1, len(history.history['loss']) + 1)  # 获取所有的 epoch 索引
    spacing = max(1, len(epochs) // 10)  # 根据 epoch 的数量调整标签的间隔

    plt.figure(figsize=(12, 5))

    # 绘制 Loss 图
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history.history['loss'], label='Training Loss')
    plt.plot(epochs, history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.xticks(epochs[::spacing])  # 设置 x 轴显示所有的 epoch，每隔 spacing 个显示一个标签
    plt.legend()
    plt.title('Training and Validation Loss')

    # 绘制 Accuracy 图
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history.history['accuracy'], label='Training Accuracy')
    plt.plot(epochs, history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.xticks(epochs[::spacing])  # 设置 x 轴显示所有的 epoch，每隔 spacing 个显示一个标签
    plt.legend()
    plt.title('Training and Validation Accuracy')

    plt.show()

def load_npy_data(npy_path, gesture_map, num_frames=35, train_ratio=0.8, val_ratio=0.1):
    data = np.load(npy_path)  # 加载 .npy 文件中的数据
    num_samples = data.shape[-1] // num_frames  # 计算样本数量

    # 按照 num_frames 切割数据
    X_segments = [data[..., i*num_frames:(i+1)*num_frames] for i in range(num_samples)]
    X_segments = np.stack(X_segments, axis=0)

    # 创建对应的标签（假设标签为最后一个类别）
    y_segments = np.full((num_samples,), fill_value=len(gesture_map)-1)  # 标签为最后一个类别

    # 划分训练、验证、测试集
    num_train = int(len(X_segments) * train_ratio)
    num_val = int(len(X_segments) * val_ratio)
    num_test = len(X_segments) - num_train - num_val

    X_train = X_segments[:num_train]
    y_train = y_segments[:num_train]
    X_val = X_segments[num_train:num_train + num_val]
    y_val = y_segments[num_train:num_train + num_val]
    X_test = X_segments[num_train + num_val:]
    y_test = y_segments[num_train + num_val:]

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

# 主程序
def main():
    current_dir = os.path.dirname(__file__)
    h5_dir = os.path.join(current_dir, 'training_set')
    num_frames = 35
    (X_train, y_train), (X_val, y_val), (X_test, y_test), gesture_map = load_data(h5_dir, num_frames=num_frames)

    # 加载 無手勢背景 .npy 文件并将数据添加到训练、验证和测试集
    npy_path = os.path.join(current_dir, 'captured_frames.npy')
    (X_train_npy, y_train_npy), (X_val_npy, y_val_npy), (X_test_npy, y_test_npy) = load_npy_data(npy_path, gesture_map, num_frames=num_frames)

    # 合并 HDF5 数据和 .npy 数据
    X_train = np.concatenate((X_train, X_train_npy), axis=0)
    y_train = np.concatenate((y_train, y_train_npy), axis=0)
    X_val = np.concatenate((X_val, X_val_npy), axis=0)
    y_val = np.concatenate((y_val, y_val_npy), axis=0)
    X_test = np.concatenate((X_test, X_test_npy), axis=0)
    y_test = np.concatenate((y_test, y_test_npy), axis=0)

    num_classes = len(gesture_map)

    # 训练数据标准化
    X_train, y_train, mean, std = preprocess_data(X_train, y_train, num_classes, sav=True)

    # 使用训练数据的标准化参数处理验证和测试数据
    X_val, y_val, _, _ = preprocess_data(X_val, y_val, num_classes, mean, std)
    X_test, y_test, _, _ = preprocess_data(X_test, y_test, num_classes, mean, std)

    # number_sample, depth, height, width, frame
    num_samples, depth_train, height_train, width_train, num_frames = X_train.shape

    # input_shape = (35, 32, 32, 2) # frames, height, width, channels
    input_shape = (num_frames, height_train, width_train, depth_train)

    # 转置输入数据轴顺序以匹配模型的期望输入形状
    X_train = np.transpose(X_train, (0, 4, 2, 3, 1))
    X_val = np.transpose(X_val, (0, 4, 2, 3, 1))
    X_test = np.transpose(X_test, (0, 4, 2, 3, 1))

    model = create_model(input_shape, num_classes)
    print(model.summary())
    history = train_model(model, X_train, y_train, X_val, y_val)

    evaluate_model(model, X_test, y_test, gesture_map)

    trained_model_dir = os.path.join(current_dir, 'trained_model')
    if not os.path.exists(trained_model_dir):
        os.makedirs(trained_model_dir)
    model.save(os.path.join(trained_model_dir, 'gesture_recognition_model.h5'))

    plot_training_history(history)

if __name__ == '__main__':
    main()


