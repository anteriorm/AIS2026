import numpy as np
import matplotlib.pyplot as plt
import time
import random
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from typing import List, Tuple, Dict


def generate_cities(n_cities: int, config: str = 'uniform',
                    x_range=(0, 100), y_range=(0, 100),
                    centers=None, cluster_std=15.0) -> List[Tuple[float, float]]:
    if config == 'uniform':
        return [(random.uniform(*x_range), random.uniform(*y_range)) for _ in range(n_cities)]
    elif config == 'blobs':
        if centers is None:
            centers = [(20, 20), (80, 80), (50, 50)]
        X, _ = make_blobs(n_samples=n_cities, centers=centers, cluster_std=cluster_std, random_state=42)
        return [(x, y) for x, y in X]
    elif config == 'gaussian':
        mean = ((x_range[0] + x_range[1]) / 2, (y_range[0] + y_range[1]) / 2)
        cov = [[(x_range[1] - x_range[0]) / 4, 0], [0, (y_range[1] - y_range[0]) / 4]]
        points = np.random.multivariate_normal(mean, cov, n_cities)
        return [(x, y) for x, y in points]
    else:
        raise ValueError("Неизвестная конфигурация")


def distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def compute_wcss(cities, labels, centroids):
    wcss = 0.0
    cities = np.array(cities)
    for i, city in enumerate(cities):
        centroid = centroids[labels[i]]
        wcss += distance(city, centroid) ** 2
    return wcss


def greedy_centroid_selection(cities, K):
    cities = list(cities)
    n_cities = len(cities)

    selected_indices = [random.randint(0, n_cities - 1)]
    centroids = [cities[selected_indices[0]]]

    for _ in range(1, K):
        min_distances = []
        for i, city in enumerate(cities):
            if i in selected_indices:
                min_distances.append(-1)
                continue
            min_dist = min(distance(city, cent) for cent in centroids)
            min_distances.append(min_dist)

        best_idx = np.argmax(min_distances)
        selected_indices.append(best_idx)
        centroids.append(cities[best_idx])

    return np.array(centroids)


def greedy_clustering(cities, K):
    centroids = greedy_centroid_selection(cities, K)
    labels = []
    for city in cities:
        dists = [distance(city, c) for c in centroids]
        labels.append(np.argmin(dists))
    return np.array(labels), np.array(centroids)


def kmeans_manual(cities, K, max_iters=100, tol=1e-4):
    centroids = np.array(random.sample(cities, K))
    cities = np.array(cities)
    labels = np.zeros(len(cities), dtype=int)
    iterations = 0
    for _ in range(max_iters):
        iterations += 1
        for i, city in enumerate(cities):
            dists = [distance(city, c) for c in centroids]
            labels[i] = np.argmin(dists)
        new_centroids = []
        for k in range(K):
            cluster_points = cities[labels == k]
            if len(cluster_points) > 0:
                new_centroids.append(np.mean(cluster_points, axis=0))
            else:
                new_centroids.append(centroids[k])
        new_centroids = np.array(new_centroids)
        if np.allclose(new_centroids, centroids, atol=tol):
            break
        centroids = new_centroids
    return labels, centroids, iterations


def kmeans_sklearn(cities, K):
    cities = np.array(cities)
    kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
    labels = kmeans.fit_predict(cities)
    centroids = kmeans.cluster_centers_
    wcss = kmeans.inertia_
    iterations = kmeans.n_iter_
    return labels, centroids, wcss, iterations


def elbow_method(cities, max_k=10, n_runs=5):
    cities_array = np.array(cities)
    k_range = range(1, max_k + 1)
    wcss_avg = []
    wcss_std = []

    print("\nВычисление метода локтя...")
    for k in k_range:
        wcss_values = []
        for _ in range(n_runs):
            labels, centroids, _ = kmeans_manual(cities, k)
            wcss = compute_wcss(cities, labels, centroids)
            wcss_values.append(wcss)
        avg = np.mean(wcss_values)
        std = np.std(wcss_values)
        wcss_avg.append(avg)
        wcss_std.append(std)
        print(f"K={k}: WCSS = {avg:.2f} (±{std:.2f})")

    plt.figure(figsize=(10, 6))
    plt.plot(k_range, wcss_avg, 'bo-', linewidth=2, markersize=8, label='Средний WCSS')
    plt.fill_between(k_range,
                     [wcss_avg[i] - wcss_std[i] for i in range(len(wcss_avg))],
                     [wcss_avg[i] + wcss_std[i] for i in range(len(wcss_avg))],
                     alpha=0.2, color='blue', label='Стандартное отклонение')
    plt.xlabel('Количество кластеров (K)', fontsize=12)
    plt.ylabel('WCSS (сумма квадратов расстояний)', fontsize=12)
    plt.title('Метод локтя для выбора оптимального K (ручная реализация K-means)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def run_experiment(cities, K):
    results = {}

    start = time.perf_counter()
    labels_greedy, centroids_greedy = greedy_clustering(cities, K)
    time_greedy = time.perf_counter() - start
    wcss_greedy = compute_wcss(cities, labels_greedy, centroids_greedy)
    results['Жадный алгоритм'] = {
        'time': time_greedy,
        'wcss': wcss_greedy,
        'labels': labels_greedy,
        'centroids': centroids_greedy,
        'iterations': None
    }

    start = time.perf_counter()
    labels_manual, centroids_manual, iters_manual = kmeans_manual(cities, K)
    time_manual = time.perf_counter() - start
    wcss_manual = compute_wcss(cities, labels_manual, centroids_manual)
    results['Ручной K-средних'] = {
        'time': time_manual,
        'wcss': wcss_manual,
        'labels': labels_manual,
        'centroids': centroids_manual,
        'iterations': iters_manual
    }

    start = time.perf_counter()
    labels_sklearn, centroids_sklearn, wcss_sklearn, iters_sklearn = kmeans_sklearn(cities, K)
    time_sklearn = time.perf_counter() - start
    results['K-средних (sklearn)'] = {
        'time': time_sklearn,
        'wcss': wcss_sklearn,
        'labels': labels_sklearn,
        'centroids': centroids_sklearn,
        'iterations': iters_sklearn
    }

    return results

def plot_clusters(cities, results, K, title_prefix=""):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, K))

    method_names_ru = {
        'Жадный алгоритм': 'Жадный алгоритм (один проход)',
        'Ручной K-средних': 'Ручная реализация K-средних',
        'K-средних (sklearn)': 'K-средних (библиотечная)'
    }

    for ax, (method, data) in zip(axes, results.items()):
        method_name = method_names_ru.get(method, method)
        iters_text = f", итераций={data['iterations']}" if data['iterations'] is not None else ""
        ax.set_title(f"{method_name}\nWCSS={data['wcss']:.2f}, время={data['time']:.4f}с{iters_text}")

        for i in range(K):
            pts = [cities[j] for j in range(len(cities)) if data['labels'][j] == i]
            if pts:
                xs, ys = zip(*pts)
                ax.scatter(xs, ys, color=colors[i], alpha=0.6, label=f'Кластер {i + 1}')

        ax.scatter(data['centroids'][:, 0], data['centroids'][:, 1],
                   c='black', marker='X', s=200, label='Центроиды')
        ax.set_xlabel("Координата X")
        ax.set_ylabel("Координата Y")
        ax.legend()

    plt.suptitle(title_prefix, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def test_all_configurations():
    configs = [
        ('равномерное', 'uniform', {}),
        ('гауссово облако', 'gaussian', {}),
        ('сгустки (3 центра, разброс 10)', 'blobs', {'centers': [(20, 20), (80, 80), (50, 50)], 'cluster_std': 10}),
        ('сгустки (3 центра, разброс 5)', 'blobs', {'centers': [(10, 10), (90, 90), (50, 50)], 'cluster_std': 5}),
        ('сгустки (2 центра, разброс 15)', 'blobs', {'centers': [(30, 30), (70, 70)], 'cluster_std': 15}),
    ]
    n_cities_list = [100, 500]
    K_list = [3, 5, 8]

    print("=" * 80)
    print("Результаты экспериментов (сравнение трёх методов)")
    print("=" * 80)

    for n_cities in n_cities_list:
        for K in K_list:
            for config_name, config_type, params in configs:
                cities = generate_cities(n_cities, config=config_type, **params)
                print(f"\nКонфигурация: {config_name} | городов={n_cities} | K={K}")
                print("-" * 50)
                results = run_experiment(cities, K)
                for method, data in results.items():
                    iters = f" (итераций={data['iterations']})" if data['iterations'] is not None else ""
                    print(f"{method:25} | время: {data['time']:.6f} с | WCSS: {data['wcss']:.2f}{iters}")

                title = f"Кластеризация: {config_name}, городов={n_cities}, K={K}"
                plot_clusters(cities, results, K, title_prefix=title)


def main():
    mode = input("Выберите режим: 1 - ручной ввод, 2 - автоматический тест: ")
    if mode == '1':
        try:
            n_cities = int(input("Введите количество городов: "))
            K = int(input("Введите количество кластеров K: "))
        except ValueError:
            print("Ошибка ввода")
            return

        print("\nВыберите конфигурацию городов:")
        print("1 - Равномерное распределение")
        print("2 - Гауссово облако")
        print("3 - Сгустки (3 центра)")
        config_choice = input("Ваш выбор (1/2/3): ")
        if config_choice == '1':
            cities = generate_cities(n_cities, config='uniform')
            config_name = "Равномерное распределение"
        elif config_choice == '2':
            cities = generate_cities(n_cities, config='gaussian')
            config_name = "Гауссово облако"
        elif config_choice == '3':
            cities = generate_cities(n_cities, config='blobs')
            config_name = "Сгустки (3 центра)"
        else:
            cities = generate_cities(n_cities, config='uniform')
            config_name = "Равномерное распределение"

        elbow = input("Построить график метода локтя для выбора K? (y/n): ").lower()
        if elbow == 'y':
            max_k = min(15, n_cities)
            elbow_method(cities, max_k=max_k, n_runs=5)

        results = run_experiment(cities, K)
        print("\nРезультаты:")
        print("-" * 50)
        for method, data in results.items():
            iters = f" (итераций={data['iterations']})" if data['iterations'] is not None else ""
            print(f"{method:25} | время: {data['time']:.6f} с | WCSS: {data['wcss']:.2f}{iters}")

        title = f"Кластеризация: {config_name}, городов={n_cities}, K={K}"
        plot_clusters(cities, results, K, title_prefix=title)
    else:
        test_all_configurations()


if __name__ == "__main__":
    main()