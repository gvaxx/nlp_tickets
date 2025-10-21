# =============================================================================
# КЛАСС ДЛЯ КЛАСТЕРИЗАЦИИ ТЕКСТОВ
# =============================================================================

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from umap.umap_ import UMAP
import hdbscan
import matplotlib.pyplot as plt
from typing import Optional, Tuple
import random

class TextClusteringPipeline:
    """
    Полный pipeline для кластеризации текстов:
    1. Создание эмбеддингов
    2. Уменьшение размерности (UMAP)
    3. Кластеризация (HDBSCAN)
    
    Вход: DataFrame с колонками [id, text]
    Выход: DataFrame с колонками [id, text, cluster_id, cluster_confidence]
    """
    
    def __init__(
        self,
        embedding_model_name: str = 'ai-forever/FRIDA',
        random_seed: int = 42
    ):
        """
        Инициализация pipeline
        
        Args:
            embedding_model_name: название модели для эмбеддингов
            random_seed: сид для воспроизводимости
        """
        # Устанавливаем seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        self.random_seed = random_seed
        self.embedding_model_name = embedding_model_name
        
        # Модели (инициализируются при использовании)
        self.embedding_model = None
        self.umap_model = None
        self.clusterer = None
        
        # Данные
        self.embeddings = None
        self.embeddings_reduced = None
        self.cluster_labels = None
        self.cluster_probabilities = None
    
    def _load_embedding_model(self):
        """Загружает модель для эмбеддингов"""
        if self.embedding_model is None:
            print(f"📥 Загружаем модель эмбеддингов: {self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
    
    def create_embeddings(
        self,
        texts: list,
        batch_size: int = 32,
        use_multiprocessing: bool = True
    ) -> np.ndarray:
        """
        Создание эмбеддингов для текстов
        
        Args:
            texts: список текстов
            batch_size: размер батча
            use_multiprocessing: использовать ли мультипроцессинг
        
        Returns:
            массив эмбеддингов (n_samples, embedding_dim)
        """
        self._load_embedding_model()
        
        print(f"🔄 Создание эмбеддингов для {len(texts)} текстов...")
        
        if use_multiprocessing:
            import multiprocessing
            num_workers = multiprocessing.cpu_count()
            print(f"   Используем {num_workers} CPU ядер")
            
            pool = self.embedding_model.start_multi_process_pool()
            
            embeddings = self.embedding_model.encode(
                texts,
                batch_size=batch_size,
                pool=pool,
                chunk_size=1000,
                show_progress_bar=True
            )
            
            self.embedding_model.stop_multi_process_pool(pool)
        else:
            embeddings = self.embedding_model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True
            )
        
        self.embeddings = embeddings
        print(f"✅ Размерность эмбеддингов: {embeddings.shape}")
        
        return embeddings
    
    def reduce_dimensions(
        self,
        embeddings: Optional[np.ndarray] = None,
        n_components: int = 5,
        n_neighbors: int = 15,
        min_dist: float = 0.0,
        metric: str = 'cosine'
    ) -> np.ndarray:
        """
        Уменьшение размерности эмбеддингов с помощью UMAP
        
        Args:
            embeddings: массив эмбеддингов (если None, использует self.embeddings)
            n_components: целевая размерность
            n_neighbors: количество соседей (баланс локальной/глобальной структуры)
            min_dist: минимальное расстояние между точками
            metric: метрика расстояния
        
        Returns:
            массив уменьшенной размерности (n_samples, n_components)
        """
        if embeddings is None:
            if self.embeddings is None:
                raise ValueError("Сначала создайте эмбеддинги через create_embeddings()")
            embeddings = self.embeddings
        
        print(f"🔄 Уменьшение размерности с UMAP до {n_components}D...")
        
        self.umap_model = UMAP(
            n_neighbors=n_neighbors,
            n_components=n_components,
            min_dist=min_dist,
            metric=metric,
            random_state=self.random_seed
        )
        
        embeddings_reduced = self.umap_model.fit_transform(embeddings)
        
        self.embeddings_reduced = embeddings_reduced
        print(f"✅ Размерность после UMAP: {embeddings_reduced.shape}")
        
        return embeddings_reduced
    
    def cluster(
        self,
        embeddings: Optional[np.ndarray] = None,
        min_cluster_size: int = 50,
        min_samples: int = 10,
        metric: str = 'euclidean',
        cluster_selection_method: str = 'eom'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Кластеризация с помощью HDBSCAN
        
        Args:
            embeddings: массив эмбеддингов (если None, использует self.embeddings_reduced)
            min_cluster_size: минимальный размер кластера
            min_samples: минимальное количество соседей для core point
            metric: метрика расстояния
            cluster_selection_method: метод выбора кластеров ('eom' или 'leaf')
        
        Returns:
            (cluster_labels, cluster_probabilities)
        """
        if embeddings is None:
            if self.embeddings_reduced is None:
                raise ValueError("Сначала примените reduce_dimensions()")
            embeddings = self.embeddings_reduced
        
        print(f"🔄 Кластеризация с HDBSCAN...")
        
        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric=metric,
            cluster_selection_method=cluster_selection_method,
            prediction_data=True
        )
        
        cluster_labels = self.clusterer.fit_predict(embeddings)
        cluster_probabilities = self.clusterer.probabilities_
        
        self.cluster_labels = cluster_labels
        self.cluster_probabilities = cluster_probabilities
        
        # Статистика
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        
        print(f"✅ Найдено кластеров: {n_clusters}")
        print(f"   Шумовых точек: {n_noise} ({n_noise/len(cluster_labels)*100:.2f}%)")
        
        # Размеры кластеров
        unique, counts = np.unique(cluster_labels, return_counts=True)
        print("\n📊 Размеры кластеров:")
        for cluster_id, count in sorted(zip(unique, counts)):
            if cluster_id == -1:
                print(f"   Шум: {count}")
            else:
                print(f"   Кластер {cluster_id}: {count}")
        
        return cluster_labels, cluster_probabilities
    
    def fit(
        self,
        df: pd.DataFrame,
        id_col: str = 'id',
        text_col: str = 'text',
        # Параметры эмбеддингов
        batch_size: int = 32,
        use_multiprocessing: bool = True,
        # Параметры UMAP
        n_components: int = 5,
        n_neighbors: int = 15,
        # Параметры HDBSCAN
        min_cluster_size: int = 50,
        min_samples: int = 10
    ) -> pd.DataFrame:
        """
        Полный pipeline: эмбеддинги → UMAP → HDBSCAN
        
        Args:
            df: входной DataFrame с колонками [id, text]
            id_col: название колонки с ID
            text_col: название колонки с текстом
            остальные: параметры для каждого шага
        
        Returns:
            DataFrame с колонками [id, text, cluster_id, cluster_confidence]
        """
        # Проверяем входные данные
        if id_col not in df.columns or text_col not in df.columns:
            raise ValueError(f"DataFrame должен содержать колонки '{id_col}' и '{text_col}'")
        
        print("="*80)
        print("ЗАПУСК PIPELINE КЛАСТЕРИЗАЦИИ")
        print("="*80)
        print(f"Количество текстов: {len(df)}\n")
        
        # 1. Эмбеддинги
        texts = df[text_col].tolist()
        self.create_embeddings(
            texts,
            batch_size=batch_size,
            use_multiprocessing=use_multiprocessing
        )
        
        # 2. Уменьшение размерности
        self.reduce_dimensions(
            n_components=n_components,
            n_neighbors=n_neighbors
        )
        
        # 3. Кластеризация
        self.cluster(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples
        )
        
        # 4. Формируем результат
        result_df = df[[id_col, text_col]].copy()
        result_df['cluster_id'] = self.cluster_labels
        result_df['cluster_confidence'] = self.cluster_probabilities
        
        # Сортируем по кластерам
        result_df = result_df.sort_values('cluster_id').reset_index(drop=True)
        
        print("\n" + "="*80)
        print("✅ PIPELINE ЗАВЕРШЁН")
        print("="*80)
        
        return result_df
    
    def predict(
        self,
        new_texts: list
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Предсказание кластера для новых текстов
        
        Args:
            new_texts: список новых текстов
        
        Returns:
            (cluster_labels, cluster_strengths)
        """
        if self.embedding_model is None or self.umap_model is None or self.clusterer is None:
            raise ValueError("Сначала обучите модель через fit()")
        
        print(f"🔮 Предсказание кластеров для {len(new_texts)} новых текстов...")
        
        # 1. Эмбеддинги
        new_embeddings = self.embedding_model.encode(
            new_texts,
            convert_to_numpy=True,
            show_progress_bar=True
        )
        
        # 2. UMAP
        new_embeddings_reduced = self.umap_model.transform(new_embeddings)
        
        # 3. Предсказание
        labels, strengths = hdbscan.approximate_predict(
            self.clusterer,
            new_embeddings_reduced
        )
        
        print(f"✅ Предсказание завершено")
        
        return labels, strengths
    
    def visualize(
        self,
        df: pd.DataFrame,
        save_path: str = 'clustering_visualization.png',
        figsize: Tuple[int, int] = (15, 10)
    ):
        """
        Визуализация кластеров в 2D
        
        Args:
            df: DataFrame с результатами (должен содержать 'cluster_id')
            save_path: путь для сохранения изображения
            figsize: размер фигуры
        """
        if self.embeddings is None:
            raise ValueError("Сначала обучите модель через fit()")
        
        print("🎨 Создание визуализации...")
        
        # UMAP в 2D для визуализации
        umap_2d = UMAP(
            n_neighbors=15,
            n_components=2,
            min_dist=0.0,
            metric='cosine',
            random_state=self.random_seed
        )
        embeddings_2d = umap_2d.fit_transform(self.embeddings)
        
        # График
        plt.figure(figsize=figsize)
        
        cluster_labels = df['cluster_id'].values
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        
        scatter = plt.scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            c=cluster_labels,
            cmap='Spectral',
            s=5,
            alpha=0.6
        )
        
        plt.colorbar(scatter, label='Cluster ID')
        plt.title(
            f'Кластеризация текстов\n{n_clusters} кластеров, {n_noise} outliers',
            fontsize=14
        )
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        print(f"✅ Визуализация сохранена в '{save_path}'")
        plt.show()
    
    def get_cluster_samples(
        self,
        df: pd.DataFrame,
        n_samples: int = 5,
        text_col: str = 'text'
    ):
        """
        Выводит примеры текстов из каждого кластера
        
        Args:
            df: DataFrame с результатами
            n_samples: количество примеров на кластер
            text_col: название колонки с текстом
        """
        print("\n" + "="*80)
        print("ПРИМЕРЫ ТЕКСТОВ ИЗ КЛАСТЕРОВ")
        print("="*80)
        
        for cluster_id in sorted(df['cluster_id'].unique()):
            if cluster_id == -1:
                cluster_name = "ШУМ"
            else:
                cluster_name = f"КЛАСТЕР {cluster_id}"
            
            cluster_df = df[df['cluster_id'] == cluster_id]
            
            print(f"\n📁 {cluster_name} ({len(cluster_df)} текстов):")
            
            samples = cluster_df.head(n_samples)
            for i, (_, row) in enumerate(samples.iterrows(), 1):
                text = row[text_col]
                confidence = row.get('cluster_confidence', 0)
                
                # Обрезаем длинные тексты
                text_short = text[:150] + "..." if len(text) > 150 else text
                print(f"  {i}. [{confidence:.2f}] {text_short}")
    
    def save_results(
        self,
        df: pd.DataFrame,
        csv_path: str = 'clustering_results.csv',
        excel_path: Optional[str] = None
    ):
        """
        Сохранение результатов
        
        Args:
            df: DataFrame с результатами
            csv_path: путь для CSV файла
            excel_path: путь для Excel файла (если нужен)
        """
        # CSV
        df.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"✅ Результаты сохранены в '{csv_path}'")
        
        # Excel (опционально)
        if excel_path:
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                # Все данные
                df.to_excel(writer, sheet_name='Все данные', index=False)
                
                # Отдельные листы для кластеров
                for cluster_id in sorted(df['cluster_id'].unique()):
                    cluster_df = df[df['cluster_id'] == cluster_id]
                    
                    if cluster_id == -1:
                        sheet_name = 'Шум'
                    else:
                        sheet_name = f'Кластер {cluster_id}'
                    
                    cluster_df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
                
                # Статистика
                stats = df.groupby('cluster_id').agg({
                    'cluster_id': 'count',
                    'cluster_confidence': 'mean'
                }).rename(columns={
                    'cluster_id': 'Количество',
                    'cluster_confidence': 'Средняя уверенность'
                })
                stats.to_excel(writer, sheet_name='Статистика')
            
            print(f"✅ Результаты сохранены в '{excel_path}'")


# =============================================================================
# ПРИМЕР ИСПОЛЬЗОВАНИЯ
# =============================================================================

if __name__ == "__main__":
    # Тестовые данные
    data = {
        'id': list(range(1, 9)),
        'text': [
            'Исправить баг с авторизацией пользователя в системе',
            'Проблема с входом через OAuth требует исправления',
            'Добавить кнопку экспорта данных в CSV формат',
            'Реализовать функцию выгрузки отчётов в Excel',
            'Оптимизировать SQL запрос для отчёта по продажам',
            'Ускорить загрузку дашборда с большими данными',
            'Обновить документацию API для новых endpoints',
            'Написать README с инструкцией по установке'
        ]
    }
    
    df_input = pd.DataFrame(data)
    
    # Или загрузи из файла:
    # df_input = pd.read_csv('tasks.csv')
    # df_input = pd.read_excel('tasks.xlsx')
    
    # Создаём pipeline
    # 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2' - среднее качество, быстро
    # 'ai-forever/FRIDA' - высокое качество, но медленно
    pipeline = TextClusteringPipeline(
        embedding_model_name='ai-forever/FRIDA'
    )
    
    # Обучаем (fit)
    df_result = pipeline.fit(
        df_input,
        id_col='id',
        text_col='text',
        # Параметры можно настроить:
        min_cluster_size=2,  # для маленького датасета
        min_samples=1,
        n_components=5
    )
    
    # Просмотр примеров
    pipeline.get_cluster_samples(df_result, n_samples=3)
    
    # Визуализация
    pipeline.visualize(df_result)
    
    # Сохранение
    pipeline.save_results(
        df_result,
        csv_path='results.csv',
        excel_path='results.xlsx'
    )
    
    # Предсказание для новых текстов
    new_tasks = [
        'Починить проблему с логином пользователя',
        'Добавить экспорт в JSON формат'
    ]
    labels, confidences = pipeline.predict(new_tasks)
    
    print("\n🔮 Предсказания для новых задач:")
    for task, label, conf in zip(new_tasks, labels, confidences):
        print(f"  '{task}' → Кластер {label} (уверенность: {conf:.2f})")