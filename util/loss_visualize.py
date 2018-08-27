import seaborn as sns;

sns.set()
import matplotlib.pyplot as plt
import pandas as pd

loss_file = "../tmp/loss.txt"

lines = open(loss_file).readlines()
lines = [line for line in lines if "loss" in line]


def extract_line(line):
    segs = line.split("-")
    train_loss = float(segs[2].split(":")[1][1:])
    val_loss = float(segs[4].split(":")[1][1:])
    return train_loss, val_loss


lines = [extract_line(line) for line in lines]
train_loss, val_loss = list(zip(*lines))
steps = list(range(1, len(train_loss) + 1))
data = pd.DataFrame({
    "step": steps,
    "train_loss": train_loss,
    "val_loss": val_loss})
fmri = sns.load_dataset("fmri")
# ax = sns.lineplot(x="timepoint", y="signal", data=fmri)
ax = sns.lineplot(x="step", y="train_loss", data=data)
sns.lineplot(x="step", y="val_loss", data=data)
plt.show()
print(0)
