from os import W_OK
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

#3期間モデル、中年期の労働所得が若年期の労働所得に依存している場合

#効用関数を書いている
def util(cons, gamma):
  return max(cons, 1e-4)**(1.0-gamma)/(1.0-gamma)


# パラメーターの設定
gamma = 2.0 #リスク回避度（高い）
beta = 0.985**20 #将来に対するウェイト
r = 1.025**20-1.0 #利子率
y = np.array([1.0, 1.2, 0.4])
JJ = 3 #期間数
l = np.array([0.8027, 1.0, 1.2457])
NL = 3
prob = np.array([
    [0.7451, 0.2528, 0.0021],
    [0.1360, 0.7281, 0.1361],
    [0.0021, 0.2528, 0.7451]
])
mu_1 = np.array([1.0/NL, 1.0/NL, 1.0/NL])
mu_2 = np.zeros(NL)

for il in range(NL):
    for ilp in range(NL):
        mu_2[ilp] += prob[il, ilp]*mu_1[il]


# グリッド
a_l = 0.0
a_u = 2.0
NA = 100
a = np.linspace(a_l, a_u, NA)

# initialization　(outputを入れる箱を用意している)
v = np.zeros((JJ, NA, NL))
iaplus = np.zeros((JJ, NA, NL), dtype=int)
aplus = np.zeros((JJ, NA, NL))

# 年金（総税収）の計算
total_tax = 0.0
for il in range(NL):
    total_tax += mu_2[il] * l[il] * 0.3  # 中年期の所得 × 税率 × 人口比率

# 年金給付額（税収をrで老年期まで運用 → 全人口で均等割） p = pension
p = (1 + r) * total_tax

# backward induction、つまり、3期間目・老年期から解くことによって、２期間目、１期間目がわかる

# 老年期　 y[2]（労働所得がなくなる）→ 年金(pension)に置き換えしました
for ia in range(NA):
    v[2, ia, :] = util(p + (1.0+r)*a[ia], gamma)


# 中年期
for il in range(NL):
    for ia in range(NA):
        reward = np.zeros(NA)
        for iap in range(NA):
            reward[iap] = util(l[il] + (1.0+r)*a[ia] - a[iap], gamma) + beta*v[2, iap, 0]
        iaplus[1, ia, il] = np.argmax(reward)
        aplus[1, ia, il] = a[iaplus[1, ia, il]]
        v[1, ia, il] = reward[iaplus[1, ia, il]]

# 若年期
for il in range(NL):
    for ia in range(NA):
        reward = np.zeros(NA)
        for iap in range(NA):

            EV = 0.0
            for ilp in range(NL):
                EV += prob[il, ilp]*v[1, iap, ilp]

            reward[iap] = util(l[il] + (1.0+r)*a[ia] - a[iap], gamma) + beta*EV

        iaplus[0, ia, il] = np.argmax(reward)
        aplus[0, ia, il] = a[iaplus[0, ia, il]]
        v[0, ia, il] = reward[iaplus[0, ia, il]]


#総税収、一人当たりの年金
print(total_tax)
print(p)