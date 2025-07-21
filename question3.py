import numpy as np
import matplotlib.pyplot as plt

# パラメータ設定
gamma = 2.0
beta = 0.985**20
r = 1.025**20 - 1.0
y = np.array([1.0, 1.2, 0.4])
l = np.array([0.8027, 1.0, 1.2457])
prob = np.array([
    [0.7451, 0.2528, 0.0021],
    [0.1360, 0.7281, 0.1361],
    [0.0021, 0.2528, 0.7451]
])
NL = 3
mu_1 = np.array([1.0/NL]*NL)
mu_2 = np.zeros(NL)
for il in range(NL):
    for ilp in range(NL):
        mu_2[ilp] += prob[il, ilp] * mu_1[il]

# 効用関数
def util(cons, gamma):
    return max(cons, 1e-4)**(1.0 - gamma) / (1.0 - gamma)

# グリッド
a_l, a_u, NA = 0.0, 2.0, 100
a = np.linspace(a_l, a_u, NA)
JJ = 3  # 3期間

# 年金なしの計算 (pension=noの場合)
v_no = np.zeros((JJ, NA, NL))
aplus_no = np.zeros((JJ, NA, NL))

for ia in range(NA):
    v_no[2, ia, :] = util(y[2] + (1 + r)*a[ia], gamma)

for il in range(NL):
    for ia in range(NA):
        rewards = [util(l[il] + (1 + r)*a[ia] - a[iap], gamma) + beta * v_no[2, iap, 0] for iap in range(NA)]
        best = np.argmax(rewards)
        aplus_no[1, ia, il] = a[best]
        v_no[1, ia, il] = rewards[best]

for il in range(NL):
    for ia in range(NA):
        rewards = []
        for iap in range(NA):
            EV = sum(prob[il, ilp] * v_no[1, iap, ilp] for ilp in range(NL))
            rewards.append(util(l[il] + (1 + r)*a[ia] - a[iap], gamma) + beta * EV)
        best = np.argmax(rewards)
        aplus_no[0, ia, il] = a[best]
        v_no[0, ia, il] = rewards[best]

# 年金ありの計算 (pension=yesの場合)
total_tax = sum(mu_2[il] * l[il] * 0.3 for il in range(NL))
pension = (1 + r) * total_tax

v_yes = np.zeros((JJ, NA, NL))
aplus_yes = np.zeros((JJ, NA, NL))

for ia in range(NA):
    v_yes[2, ia, :] = util(pension + (1 + r)*a[ia], gamma)

for il in range(NL):
    for ia in range(NA):
        rewards = [util(l[il]*0.7 + (1 + r)*a[ia] - a[iap], gamma) + beta * v_yes[2, iap, 0] for iap in range(NA)]
        best = np.argmax(rewards)
        aplus_yes[1, ia, il] = a[best]
        v_yes[1, ia, il] = rewards[best]

for il in range(NL):
    for ia in range(NA):
        rewards = []
        for iap in range(NA):
            EV = sum(prob[il, ilp] * v_yes[1, iap, ilp] for ilp in range(NL))
            rewards.append(util(l[il] + (1 + r)*a[ia] - a[iap], gamma) + beta * EV)
        best = np.argmax(rewards)
        aplus_yes[0, ia, il] = a[best]
        v_yes[0, ia, il] = rewards[best]

# グラフを作成する
plt.figure(figsize=(10, 6))
for i, label in zip(range(3), ['Low', 'Mid', 'High']):
    plt.plot(a, aplus_no[0, :, i], linestyle='--', label=f'{label} (No Pension)', color='blue')
    plt.plot(a, aplus_yes[0, :, i], linestyle='-', label=f'{label} (Pension)', color='red')

plt.xlabel("Initial Asset a1")
plt.ylabel("Next Period Asset a2")
plt.title("Policy Function Comparison (Young Age)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
