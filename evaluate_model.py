import re
import matplotlib.pyplot as plt

log = """
Epoch 1/20
530/530 - accuracy: 0.6792 - loss: 0.8609 - val_accuracy: 0.6221 - val_loss: 1.0564
Epoch 2/20
530/530 - accuracy: 0.7610 - loss: 0.6593 - val_accuracy: 0.5622 - val_loss: 1.0707
Epoch 3/20
530/530 - accuracy: 0.7702 - loss: 0.6308 - val_accuracy: 0.6533 - val_loss: 0.9360
Epoch 4/20
530/530 - accuracy: 0.7765 - loss: 0.6128 - val_accuracy: 0.7689 - val_loss: 0.6317
Epoch 5/20
530/530 - accuracy: 0.7812 - loss: 0.6023 - val_accuracy: 0.4563 - val_loss: 1.3394
Epoch 6/20
530/530 - accuracy: 0.7848 - loss: 0.5914 - val_accuracy: 0.2753 - val_loss: 4.7163
Epoch 7/20
530/530 - accuracy: 0.7868 - loss: 0.5831 - val_accuracy: 0.1887 - val_loss: 3.6666
Epoch 8/20
530/530 - accuracy: 0.7896 - loss: 0.5767 - val_accuracy: 0.3615 - val_loss: 1.9497
Epoch 9/20
530/530 - accuracy: 0.7928 - loss: 0.5670 - val_accuracy: 0.6879 - val_loss: 0.8459
Epoch 10/20
530/530 - accuracy: 0.7944 - loss: 0.5623 - val_accuracy: 0.7840 - val_loss: 0.6033
Epoch 11/20
530/530 - accuracy: 0.7957 - loss: 0.5581 - val_accuracy: 0.7936 - val_loss: 0.5807
Epoch 12/20
530/530 - accuracy: 0.7976 - loss: 0.5526 - val_accuracy: 0.7772 - val_loss: 0.6259
Epoch 13/20
530/530 - accuracy: 0.7990 - loss: 0.5468 - val_accuracy: 0.6585 - val_loss: 0.8986
Epoch 14/20
530/530 - accuracy: 0.8008 - loss: 0.5438 - val_accuracy: 0.1385 - val_loss: 5.3333
Epoch 15/20
530/530 - accuracy: 0.8019 - loss: 0.5387 - val_accuracy: 0.2486 - val_loss: 1.9041
Epoch 16/20
530/530 - accuracy: 0.8035 - loss: 0.5322 - val_accuracy: 0.1140 - val_loss: 33.1472
Epoch 17/20
530/530 - accuracy: 0.8056 - loss: 0.5267 - val_accuracy: 0.4854 - val_loss: 1.3683
Epoch 18/20
530/530 - accuracy: 0.8085 - loss: 0.5206 - val_accuracy: 0.4270 - val_loss: 1.8240
Epoch 19/20
530/530 - accuracy: 0.8081 - loss: 0.5199 - val_accuracy: 0.3717 - val_loss: 1.5669
Epoch 20/20
530/530 - accuracy: 0.8083 - loss: 0.5159 - val_accuracy: 0.1453 - val_loss: 7.5445
"""

pattern = r"accuracy: ([\d.]+) - loss: ([\d.]+) - val_accuracy: ([\d.]+) - val_loss: ([\d.]+)"
matches = re.findall(pattern, log)
epochs = list(range(1, len(matches) + 1))
train_acc = [float(m[0]) for m in matches]
train_loss = [float(m[1]) for m in matches]
val_acc = [float(m[2]) for m in matches]
val_loss = [float(m[3]) for m in matches]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(epochs, train_acc, 'o-', label='Train')
ax1.plot(epochs, val_acc, 'o-', label='Val')
ax1.set_xlabel('Epoch'); ax1.set_ylabel('Accuracy')
ax1.set_title('Accuracy over Epochs'); ax1.legend(); ax1.grid(True)

ax2.plot(epochs, train_loss, 'o-', label='Train')
ax2.plot(epochs, val_loss, 'o-', label='Val')
ax2.set_xlabel('Epoch'); ax2.set_ylabel('Loss')
ax2.set_title('Loss over Epochs'); ax2.legend(); ax2.grid(True)

plt.tight_layout()
plt.savefig('figures/fig_training_curves_model0.png', dpi=150)
plt.show()
print("Done")
