import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ---------- 1. 读取数据 ----------
train_df = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")
test_df  = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")

X = train_df.drop("label", axis=1).values.astype("float32")   # (42000, 784)
y = train_df["label"].values
X_test = test_df.values.astype("float32")                      # (28000, 784)

# ---------- 2. MinMax 归一化 (在 reshape 之前做) ----------
scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)

# ---------- 3. reshape 成图像格式 (N, 28, 28, 1) ----------
X = X.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# ---------- 4. 训练集 / 验证集划分 ----------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)
print(X_train.shape, X_val.shape)

# ---------- 5. 定义 CNN ----------
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),

    tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(10, activation="softmax"),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

# ---------- 6. 训练 ----------
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy", patience=3, restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=128,
    callbacks=[early_stop],
    verbose=2,
)

val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
print(f"Validation accuracy: {val_acc:.4f}")

# ---------- 7. 预测并生成 submission.csv ----------
pred_probs = model.predict(X_test)
pred_labels = np.argmax(pred_probs, axis=1)

submission = pd.DataFrame({
    "ImageId": np.arange(1, len(pred_labels) + 1),
    "Label": pred_labels,
})
submission.to_csv("submission.csv", index=False)
print(submission.head())