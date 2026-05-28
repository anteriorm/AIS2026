import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve, auc,
                             adjusted_rand_score, normalized_mutual_info_score,
                             silhouette_score)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold

train = pd.read_csv('disease_train.csv')
test = pd.read_csv('disease_public_test.csv')
sub = pd.read_csv('disease_sample_submission.csv')

X_train = train.iloc[:, :-1].values
y_train = train.iloc[:, -1].values
X_test = test.values
y_test = sub['Y'].values

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

qda_raw = QuadraticDiscriminantAnalysis()
qda_raw.fit(X_train, y_train)
y_pred_raw = qda_raw.predict(X_test)
y_prob_raw = qda_raw.predict_proba(X_test)[:, 1]

qda_scaled = QuadraticDiscriminantAnalysis()
qda_scaled.fit(X_train_scaled, y_train)
y_pred_scaled = qda_scaled.predict(X_test_scaled)
y_prob_scaled = qda_scaled.predict_proba(X_test_scaled)[:, 1]

def calc_metrics(y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    cm = confusion_matrix(y_true, y_pred)
    return acc, prec, rec, f1, roc_auc, cm, fpr, tpr

acc_r, prec_r, rec_r, f1_r, auc_r, cm_r, fpr_r, tpr_r = calc_metrics(y_test, y_pred_raw, y_prob_raw)
acc_s, prec_s, rec_s, f1_s, auc_s, cm_s, fpr_s, tpr_s = calc_metrics(y_test, y_pred_scaled, y_prob_scaled)

print("=== sklearn QDA ===")
print(f"Сырые:    Acc={acc_r:.4f}, Prec={prec_r:.4f}, Rec={rec_r:.4f}, F1={f1_r:.4f}, AUC={auc_r:.4f}")
print(f"Масшт.:   Acc={acc_s:.4f}, Prec={prec_s:.4f}, Rec={rec_s:.4f}, F1={f1_s:.4f}, AUC={auc_s:.4f}")

class CustomQDA:
    def __init__(self):
        self.classes = None
        self.means = {}
        self.covs = {}
        self.priors = {}

    def fit(self, X, y):
        self.classes = np.unique(y)
        for c in self.classes:
            Xc = X[y == c]
            self.means[c] = np.mean(Xc, axis=0)
            self.covs[c] = np.cov(Xc, rowvar=False) + 1e-6 * np.eye(X.shape[1])
            self.priors[c] = len(Xc) / len(X)
        return self

    def predict_proba(self, X):
        probs = []
        for x in X:
            log_lik = []
            for c in self.classes:
                mean = self.means[c]
                cov = self.covs[c]
                inv_cov = np.linalg.inv(cov)
                diff = x - mean
                log_det = np.log(np.linalg.det(cov))
                log_lik_c = -0.5 * (log_det + diff @ inv_cov @ diff.T) + np.log(self.priors[c])
                log_lik.append(log_lik_c)
            log_lik = np.array(log_lik)
            log_lik -= np.max(log_lik)
            probs.append(np.exp(log_lik) / np.sum(np.exp(log_lik)))
        return np.array(probs)

    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes[np.argmax(probs, axis=1)]

custom = CustomQDA()
custom.fit(X_train_scaled, y_train)
y_pred_custom = custom.predict(X_test_scaled)
y_prob_custom = custom.predict_proba(X_test_scaled)[:, 1]

acc_c, prec_c, rec_c, f1_c, auc_c, cm_c, fpr_c, tpr_c = calc_metrics(y_test, y_pred_custom, y_prob_custom)
print(f"Ручной QDA: Acc={acc_c:.4f}, Prec={prec_c:.4f}, Rec={rec_c:.4f}, F1={f1_c:.4f}, AUC={auc_c:.4f}")

print("\n=== 1.2. Разные методы формирования train/test ===")

X_tr1, X_te1, y_tr1, y_te1 = train_test_split(X_train_scaled, y_train, test_size=0.3, random_state=42, stratify=y_train)
qda_split1 = QuadraticDiscriminantAnalysis()
qda_split1.fit(X_tr1, y_tr1)
y_pred1 = qda_split1.predict(X_te1)
acc1 = accuracy_score(y_te1, y_pred1)
print(f"1. train_test_split (test_size=0.3, random_state=42): Accuracy = {acc1:.4f}")

X_tr2, X_te2, y_tr2, y_te2 = train_test_split(X_train_scaled, y_train, test_size=0.3, random_state=123, stratify=y_train)
qda_split2 = QuadraticDiscriminantAnalysis()
qda_split2.fit(X_tr2, y_tr2)
y_pred2 = qda_split2.predict(X_te2)
acc2 = accuracy_score(y_te2, y_pred2)
print(f"2. train_test_split (test_size=0.3, random_state=123): Accuracy = {acc2:.4f}")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(QuadraticDiscriminantAnalysis(), X_train_scaled, y_train, cv=cv, scoring='accuracy')
print(f"3. 5-кратная кросс-валидация: средняя accuracy = {scores.mean():.4f} (+/- {scores.std():.4f})")

kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_train_scaled)
ari = adjusted_rand_score(y_train, kmeans_labels)
ami = normalized_mutual_info_score(y_train, kmeans_labels)
sil = silhouette_score(X_train_scaled, kmeans_labels)
print(f"\nKMeans: ARI={ari:.4f}, AMI={ami:.4f}, Silhouette={sil:.4f}")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('QuadraticDiscriminantAnalysis (sklearn QDA)', fontsize=14, fontweight='bold')

ax = axes[0,0]
ax.plot(fpr_r, tpr_r, label=f'Сырые (AUC={auc_r:.3f})', lw=2)
ax.plot(fpr_s, tpr_s, label=f'Масштабированные (AUC={auc_s:.3f})', lw=2, linestyle='--')
ax.plot([0,1],[0,1],'k--', lw=1)
ax.set_xlabel('FPR'); ax.set_ylabel('TPR'); ax.set_title('ROC-кривая')
ax.legend(); ax.grid(alpha=0.3)

ax = axes[0,1]
im = ax.imshow(cm_r, cmap='Blues')
ax.set_xticks([0,1]); ax.set_yticks([0,1])
ax.set_xticklabels(['Здоров','Болен']); ax.set_yticklabels(['Здоров','Болен'])
ax.set_title('Матрица ошибок (сырые)')
for i in range(2):
    for j in range(2):
        ax.text(j,i,str(cm_r[i,j]), ha='center', va='center', fontsize=14, fontweight='bold')
plt.colorbar(im, ax=ax)

ax = axes[1,0]
im = ax.imshow(cm_s, cmap='Blues')
ax.set_xticks([0,1]); ax.set_yticks([0,1])
ax.set_xticklabels(['Здоров','Болен']); ax.set_yticklabels(['Здоров','Болен'])
ax.set_title('Матрица ошибок (масштабированные)')
for i in range(2):
    for j in range(2):
        ax.text(j,i,str(cm_s[i,j]), ha='center', va='center', fontsize=14, fontweight='bold')
plt.colorbar(im, ax=ax)

ax = axes[1,1]
bars = ax.bar(['Сырые','Масшт.'], [acc_r, acc_s], color=['tomato','steelblue'])
ax.set_ylabel('Accuracy'); ax.set_title('Сравнение точности'); ax.set_ylim(0,1)
for bar,val in zip(bars, [acc_r, acc_s]):
    ax.text(bar.get_x()+bar.get_width()/2, val+0.02, f'{val:.4f}', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig('classification_results.png', dpi=150)
plt.show()

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train_scaled)

fig2, axes = plt.subplots(1,2, figsize=(12,5))
fig2.suptitle('Кластеризация KMeans (n_clusters=2)', fontsize=14, fontweight='bold')

ax = axes[0]
sc = ax.scatter(X_pca[:,0], X_pca[:,1], c=kmeans_labels, cmap='viridis', alpha=0.6)
ax.set_title('Предсказанные кластеры KMeans')
ax.set_xlabel(f'ГК1 ({pca.explained_variance_ratio_[0]:.1%})')
ax.set_ylabel(f'ГК2 ({pca.explained_variance_ratio_[1]:.1%})')
plt.colorbar(sc, ax=ax, label='Кластер')

ax = axes[1]
sc = ax.scatter(X_pca[:,0], X_pca[:,1], c=y_train, cmap='coolwarm', alpha=0.6)
ax.set_title('Истинные метки (0 – здоров, 1 – болен)')
ax.set_xlabel(f'ГК1 ({pca.explained_variance_ratio_[0]:.1%})')
ax.set_ylabel(f'ГК2 ({pca.explained_variance_ratio_[1]:.1%})')
plt.colorbar(sc, ax=ax, label='Класс')

plt.tight_layout()
plt.savefig('clustering_results.png', dpi=150)
plt.show()