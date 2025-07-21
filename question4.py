import numpy as np

# パラメータ
gamma = 2.0
beta = 0.985**20
r = 1.025**20 - 1.0
l = np.array([0.8027, 1.0, 1.2457])
y = np.array([1.0, 1.2, 0.4])  # 老年期に使う
prob = np.array([
    [0.7451, 0.2528, 0.0021],
    [0.1360, 0.7281, 0.1361],
    [0.0021, 0.2528, 0.7451]
])
NL = 3
mu = np.array([1/3, 1/3, 1/3])

# 効用関数
def util(c, gamma):
    return max(c, 1e-4)**(1 - gamma) / (1 - gamma)

# 資産グリッド
a_l, a_u, NA = 0.0, 2.0, 100
a = np.linspace(a_l, a_u, NA)
JJ = 3  # 3期間

# 年金額の計算
mu_1 = np.array([1/3]*3)
mu_2 = np.zeros(NL)
for il in range(NL):
    for ilp in range(NL):
        mu_2[ilp] += prob[il, ilp] * mu_1[il]

total_tax = sum(mu_2[il] * l[il] * 0.3 for il in range(NL))
pension = (1 + r) * total_tax

# 効用行列の初期化、ここにoutputを入れる
v_no = np.zeros((JJ, NA, NL))
v_yes = np.zeros((JJ, NA, NL))

#backward induction

# 老年期
for ia in range(NA):
    for il in range(NL):
        v_no[2, ia, il] = util(y[2] + (1 + r) * a[ia], gamma)
        v_yes[2, ia, il] = util(pension + (1 + r) * a[ia], gamma)

# 中年期
for il in range(NL):
    for ia in range(NA):
        v_no[1, ia, il] = max(
            util(l[il] + (1 + r)*a[ia] - a[iap], gamma) + beta * v_no[2, iap, 0]
            for iap in range(NA)
        )
        v_yes[1, ia, il] = max(
            util(l[il]*0.7 + (1 + r)*a[ia] - a[iap], gamma) + beta * v_yes[2, iap, 0]
            for iap in range(NA)
        )

# 若年期
for il in range(NL):
    for ia in range(NA):
        v_no[0, ia, il] = max(
            util(l[il] + (1 + r)*a[ia] - a[iap], gamma) + beta * sum(prob[il, ilp]*v_no[1, iap, ilp] for ilp in range(NL))
            for iap in range(NA)
        )
        v_yes[0, ia, il] = max(
            util(l[il] + (1 + r)*a[ia] - a[iap], gamma) + beta * sum(prob[il, ilp]*v_yes[1, iap, ilp] for ilp in range(NL))
            for iap in range(NA)
        )

# 初期資産ゼロ（a1 = 0）のときの効用を抽出
v0_no_pension = v_no[0, 0, :]
v0_with_pension = v_yes[0, 0, :]

# 平均期待効用（加重平均）
expected_utility_no_pension = np.dot(mu, v0_no_pension)
expected_utility_with_pension = np.dot(mu, v0_with_pension)

# 結果
print("【平均期待効用（初期資産ゼロ）】")
print(f"年金なし: {expected_utility_no_pension:.4f}")
print(f"年金あり: {expected_utility_with_pension:.4f}")