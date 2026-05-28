import time
import random
from math import sqrt
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import numpy as np

Point = Tuple[float, float]


def euclidean(a: Point, b: Point) -> float:
    return sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


class KNNClassifier:
    def __init__(self, k: int = 3):
        self.k = k

    def fit(self, X: List[Point], y: List[str]) -> "KNNClassifier":
        if len(X) != len(y):
            raise ValueError("X и y должны быть одной длины")
        self._X = X
        self._y = y
        self.classes_ = sorted(set(self._y))
        return self

    def _neighbors(self, x: Point) -> List[Tuple[float, str]]:
        dists = [(euclidean(x, xi), yi) for xi, yi in zip(self._X, self._y)]
        dists.sort(key=lambda t: t[0])
        return dists[:min(self.k, len(dists))]

    def predict_one(self, x: Point) -> str:
        neigh = self._neighbors(x)
        votes: Dict[str, int] = {}
        dist_sums: Dict[str, float] = {}
        for d, cls in neigh:
            votes[cls] = votes.get(cls, 0) + 1
            dist_sums[cls] = dist_sums.get(cls, 0.0) + d
        best = sorted(votes.keys(), key=lambda c: (-votes[c], dist_sums[c], c))[0]
        return best

    def predict(self, X: List[Point]) -> List[str]:
        return [self.predict_one(x) for x in X]


def build_dataset(include_dessert: bool = False) -> Tuple[List[str], List[Point], List[str]]:
    base = [
        ("Яблоко", 7, 7, "Фрукт"), ("салат", 2, 5, "Овощ"), ("бекон", 1, 2, "Протеин"),
        ("банан", 9, 1, "Фрукт"), ("орехи", 1, 5, "Протеин"), ("рыба", 1, 1, "Протеин"),
        ("сыр", 1, 1, "Протеин"), ("виноград", 8, 1, "Фрукт"), ("морковь", 2, 8, "Овощ"),
        ("апельсин", 6, 1, "Фрукт"),
    ]
    extended = [
        ("груша", 7, 4, "Фрукт"), ("клубника", 9, 2, "Фрукт"), ("персик", 8, 2, "Фрукт"),
        ("огурец", 2, 7, "Овощ"), ("капуста", 2, 6, "Овощ"), ("редис", 2, 7, "Овощ"),
        ("томат", 3, 3, "Овощ"), ("курица", 1, 2, "Протеин"), ("яйцо", 1, 1, "Протеин"),
        ("фасоль", 1, 4, "Протеин"), ("тофу", 1, 2, "Протеин"), ("арахис", 2, 6, "Протеин"),
    ]
    dessert = [("шоколад", 10, 1, "Десерт"), ("мороженое", 9, 1, "Десерт"), ("пирожное", 10, 2, "Десерт")]
    rows = base + extended
    if include_dessert:
        rows += dessert
    names = [r[0] for r in rows]
    X = [(float(r[1]), float(r[2])) for r in rows]
    y = [r[3] for r in rows]
    return names, X, y


def accuracy(y_true: List[str], y_pred: List[str]) -> float:
    if not y_true:
        return 0.0
    return sum(1 for a, b in zip(y_true, y_pred) if a == b) / len(y_true)


def stratified_kfold_indices(y: List[str], n_splits: int = 5, seed: int = 42):
    from collections import Counter
    min_class_size = min(Counter(y).values())
    if min_class_size < n_splits:
        n_splits = min_class_size
        print(f"Внимание: число фолдов уменьшено до {n_splits} (мало объектов в классе).")
    rng = random.Random(seed)
    by_class: Dict[str, List[int]] = {}
    for i, cls in enumerate(y):
        by_class.setdefault(cls, []).append(i)
    for cls in by_class:
        rng.shuffle(by_class[cls])
    folds: List[List[int]] = [[] for _ in range(n_splits)]
    for cls, idxs in by_class.items():
        for j, idx in enumerate(idxs):
            folds[j % n_splits].append(idx)
    result = []
    all_idx = set(range(len(y)))
    for test_idx in folds:
        train_idx = sorted(all_idx - set(test_idx))
        result.append((train_idx, sorted(test_idx)))
    return result


def cross_validate_with_confusion(X: List[Point], y: List[str], k: int, n_splits: int = 5):
    folds = stratified_kfold_indices(y, n_splits)
    all_y_true, all_y_pred = [], []
    fold_accuracies = []
    start = time.time()
    for train_idx, test_idx in folds:
        Xtr = [X[i] for i in train_idx]
        ytr = [y[i] for i in train_idx]
        Xte = [X[i] for i in test_idx]
        yte = [y[i] for i in test_idx]
        model = KNNClassifier(k=k).fit(Xtr, ytr)
        pred = model.predict(Xte)
        all_y_true.extend(yte)
        all_y_pred.extend(pred)
        fold_accuracies.append(accuracy(yte, pred))
    elapsed = time.time() - start
    mean_acc = sum(fold_accuracies) / len(fold_accuracies)
    return mean_acc, all_y_true, all_y_pred, elapsed


def confusion_matrix_print(y_true: List[str], y_pred: List[str], labels: List[str]):
    n = len(labels)
    idx = {label: i for i, label in enumerate(labels)}
    cm = [[0] * n for _ in range(n)]
    for t, p in zip(y_true, y_pred):
        cm[idx[t]][idx[p]] += 1
    print("\nМатрица ошибок:")
    print("         " + "".join(f"{l:>8}" for l in labels))
    for i, label in enumerate(labels):
        row = " ".join(f"{cm[i][j]:>8}" for j in range(n))
        print(f"{label:>8}: {row}")


def plot_decision_boundary(knn: KNNClassifier, X: List[Point], y: List[str], test_points: List[Point] = None,
                           test_labels: List[str] = None):
    X_arr = np.array(X)
    classes = sorted(set(y))
    colors = plt.cm.tab10(np.linspace(0, 1, len(classes)))
    color_dict = {cls: colors[i] for i, cls in enumerate(classes)}

    x_min, x_max = X_arr[:, 0].min() - 1, X_arr[:, 0].max() + 1
    y_min, y_max = X_arr[:, 1].min() - 1, X_arr[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_points = [(float(p[0]), float(p[1])) for p in grid]
    Z = knn.predict(grid_points)
    Z_arr = np.array([classes.index(z) for z in Z]).reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z_arr, alpha=0.3, levels=np.arange(len(classes) + 1) - 0.5, cmap='tab10')

    for cls in classes:
        mask = np.array(y) == cls
        plt.scatter(X_arr[mask, 0], X_arr[mask, 1], color=color_dict[cls], label=cls, edgecolors='k', s=80)

    if test_points:
        test_arr = np.array(test_points)
        plt.scatter(test_arr[:, 0], test_arr[:, 1], marker='X', c='red', s=200, label='Тестовые объекты',
                    edgecolors='black', linewidth=1)
        if test_labels:
            for i, (x_, y_) in enumerate(test_points):
                plt.annotate(test_labels[i], (x_, y_), xytext=(5, 5), textcoords='offset points', fontsize=9)
    plt.xlabel("Сладость")
    plt.ylabel("Хруст")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    print("===== Датасет: три класса (Фрукт, Овощ, Протеин) =====")
    names, X, y = build_dataset(include_dessert=False)
    print(f"Объектов: {len(X)}, классов: {set(y)}")

    k = 3
    acc_cv, y_true, y_pred, elapsed = cross_validate_with_confusion(X, y, k, n_splits=5)
    print(f"\nКросс-валидация (k={k}): средняя точность = {acc_cv:.3f} ({acc_cv * 100:.1f}%)")
    print(f"Время выполнения CV (кастомный k-NN): {elapsed:.4f} сек")
    confusion_matrix_print(y_true, y_pred, sorted(set(y)))

    knn_model = KNNClassifier(k=k).fit(X, y)
    test_points_3 = [(7, 3), (3, 7), (9, 2), (1, 4), (5, 5)]
    test_labels_3 = ["(7,3)", "(3,7)", "(9,2)", "(1,4)", "(5,5)"]
    plot_decision_boundary(knn_model, X, y, test_points_3, test_labels_3)

    # ---- Добавляем 4-й класс ----
    print("\n" + "=" * 60)
    print("===== Датасет: четыре класса (добавлен Десерт) =====")
    names4, X4, y4 = build_dataset(include_dessert=True)
    print(f"Объектов: {len(X4)}, классов: {set(y4)}")

    acc_cv4, y_true4, y_pred4, elapsed4 = cross_validate_with_confusion(X4, y4, k, n_splits=3)
    print(f"\nКросс-валидация (k={k}): средняя точность = {acc_cv4:.3f} ({acc_cv4 * 100:.1f}%)")
    print(f"Время выполнения CV (кастомный k-NN): {elapsed4:.4f} сек")
    confusion_matrix_print(y_true4, y_pred4, sorted(set(y4)))

    knn_model4 = KNNClassifier(k=k).fit(X4, y4)
    plot_decision_boundary(knn_model4, X4, y4, test_points_3, test_labels_3)

    try:
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.metrics import confusion_matrix as sk_cm
        import pandas as pd

        print("\n" + "=" * 60)
        print("===== Сравнение с sklearn =====")
        Xn = np.array(X4, dtype=float)
        yn = np.array(y4, dtype=str)
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        sk_model = KNeighborsClassifier(n_neighbors=k)
        start_sk = time.time()
        sk_pred = cross_val_predict(sk_model, Xn, yn, cv=skf)
        elapsed_sk = time.time() - start_sk
        sk_acc = accuracy(y4, sk_pred.tolist())
        print(f"Sklearn k-NN (k={k}): средняя точность = {sk_acc:.3f} ({sk_acc * 100:.1f}%)")
        print(f"Время выполнения CV (sklearn): {elapsed_sk:.4f} сек")
        cm_sk = sk_cm(yn, sk_pred, labels=np.unique(yn))
        print("Матрица ошибок (sklearn):")
        print(pd.DataFrame(cm_sk, index=np.unique(yn), columns=np.unique(yn)))
    except ImportError:
        print("\nБиблиотека sklearn не установлена, сравнение пропущено.")
    except Exception as e:
        print(f"\nОшибка при использовании sklearn: {e}")


if __name__ == "__main__":
    main()