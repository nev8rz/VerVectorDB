"""
VerVectorDB 示例代码
展示HNSW索引、PQ压缩、批量操作和过滤检索功能
"""
import numpy as np
from vervectordb import VerVectorDB


def demo_hnsw_index():
    """演示HNSW索引功能"""
    print("=" * 60)
    print("演示HNSW索引功能")
    print("=" * 60)
    
    db = VerVectorDB(vector_dim=128, db_path="demo_hnsw.pkl")
    
    n_vectors = 50000
    vectors = np.random.random((n_vectors, 128)).astype(np.float32)
    
    metadatas = [{"text": f"文档{i}", "category": i % 5} for i in range(n_vectors)]
    ids = db.batch_insert(vectors, metadatas)
    print(f"批量插入{n_vectors}个向量")
    
    db.build_hnsw_index()
    print("HNSW索引构建完成")
    
    query = np.random.random(128).astype(np.float32)
    results = db.hnsw_search(query, top_k=5)
    print(f"\nHNSW检索结果(top_k=5):")
    for r in results:
        print(f"  ID: {r['vector_id'][:8]}..., 相似度: {r['similarity']:.4f}")
    
    db.save()


def demo_pq_compression():
    """演示PQ压缩功能"""
    print("\n" + "=" * 60)
    print("演示PQ压缩功能")
    print("=" * 60)
    
    db = VerVectorDB(vector_dim=128, db_path="demo_pq.pkl")
    
    n_vectors = 5000
    vectors = np.random.random((n_vectors, 128)).astype(np.float32)
    db.batch_insert(vectors)
    print(f"插入{n_vectors}个向量")
    
    original_size = db.vectors.nbytes
    print(f"原始向量存储大小: {original_size / 1024 / 1024:.2f} MB")
    
    db.train_pq(sample_size=1000)
    
    db.compress()
    compressed_size = len(db.pq_codes.tobytes())
    print(f"压缩后存储大小: {compressed_size / 1024 / 1024:.2f} MB")
    print(f"压缩比: {original_size / compressed_size:.2f}x")
    
    decompressed = db.decompress()
    reconstruction_error = np.mean(np.abs(db.vectors - decompressed))
    print(f"平均重构误差: {reconstruction_error:.6f}")
    
    db.save()


def demo_batch_operations():
    """演示批量操作功能"""
    print("\n" + "=" * 60)
    print("演示批量操作功能")
    print("=" * 60)
    
    db = VerVectorDB(vector_dim=64, db_path="demo_batch.pkl")
    
    n_batch = 100
    n_vectors = 1000
    for batch_idx in range(n_batch):
        batch_vectors = np.random.random((10, 64)).astype(np.float32)
        batch_metas = [{"batch": batch_idx, "index": i} for i in range(10)]
        db.batch_insert(batch_vectors, batch_metas)
    
    print(f"共插入{len(db.vectors)}个向量")
    
    queries = np.random.random((20, 64)).astype(np.float32)
    results = db.batch_search(queries, top_k=3, method="brute_force", batch_size=10)
    print(f"批量检索{len(queries)}个查询，每个返回top_k=3")
    print(f"结果数量: {len(results)}")
    
    db.build_hnsw_index()
    hnsw_results = db.batch_search(queries, top_k=3, method="hnsw", batch_size=10)
    print(f"HNSW批量检索结果数量: {len(hnsw_results)}")
    
    db.save()


def demo_filtered_search():
    """演示过滤检索功能"""
    print("\n" + "=" * 60)
    print("演示过滤检索功能")
    print("=" * 60)
    
    db = VerVectorDB(vector_dim=64, db_path="demo_filter.pkl")
    
    categories = ["科技", "体育", "娱乐", "财经", "教育"]
    vectors = []
    metadatas = []
    
    for i in range(500):
        vectors.append(np.random.random(64).astype(np.float32))
        cat = categories[i % len(categories)]
        metadatas.append({
            "text": f"这是关于{cat}的文章{i}",
            "category": cat,
            "tags": [cat, "热门"]
        })
    
    db.batch_insert(vectors, metadatas)
    print(f"插入{len(db.vectors)}个带元数据的向量")
    
    query = np.random.random(64).astype(np.float32)
    results = db.filtered_search(query, top_k=3, keywords=["科技"])
    print(f"\n关键词过滤搜索 '科技'，结果:")
    for r in results:
        print(f"  文本: {r['metadata']['text']}, 类别: {r['metadata']['category']}")
    
    def category_filter(meta):
        return meta.get("category") == "体育"
    
    results = db.filtered_search(query, top_k=3, metadata_filter=category_filter)
    print(f"\n元数据过滤搜索 '体育'，结果:")
    for r in results:
        print(f"  文本: {r['metadata']['text']}, 类别: {r['metadata']['category']}")
    
    results = db.filtered_search(query, top_k=3, keywords=["娱乐"], 
                                  metadata_filter=lambda m: m.get("category") == "娱乐")
    print(f"\n组合过滤搜索，结果:")
    for r in results:
        print(f"  文本: {r['metadata']['text']}")
    
    db.save()


def calculate_recall(ground_truth_ids, result_ids):
    """计算召回率"""
    if not ground_truth_ids:
        return 0.0
    ground_truth_set = set(ground_truth_ids)
    hits = sum(1 for r_id in result_ids if r_id in ground_truth_set)
    return hits / len(ground_truth_set)


def demo_recall_comparison():
    """对比不同检索方法的召回率"""
    print("\n" + "=" * 60)
    print("对比不同检索方法的召回率")
    print("=" * 60)
    
    import time
    
    db = VerVectorDB(vector_dim=128, db_path="demo_recall.pkl")
    
    # 插入数据 - 增加数据量
    n_vectors = 50000  # 从5000增加到50000
    vectors = np.random.random((n_vectors, 128)).astype(np.float32)
    db.batch_insert(vectors)
    print(f"插入{n_vectors}个向量")
    
    # 构建索引
    db.build_ivf_index(n_clusters=16)
    db.build_hnsw_index()
    print("索引构建完成")
    
    # 生成测试查询
    n_queries = 100
    top_k = 10
    queries = np.random.random((n_queries, 128)).astype(np.float32)
    
    # 暴力检索结果作为基准
    print(f"\n执行{n_queries}个查询，对比召回率...")
    bf_results_list = []
    hnsw_results_list = []
    ivf_results_list = []
    
    for query in queries:
        bf_results = db.brute_force_search(query, top_k=top_k)
        bf_results_list.append([r['vector_id'] for r in bf_results])
    
    # HNSW检索
    for query in queries:
        hnsw_results = db.hnsw_search(query, top_k=top_k)
        hnsw_results_list.append([r['vector_id'] for r in hnsw_results])
    
    # IVF检索
    for query in queries:
        ivf_results = db.ivf_search(query, top_k=top_k)
        ivf_results_list.append([r['vector_id'] for r in ivf_results])
    
    # 计算召回率
    bf_recall = 1.0  # 暴力检索是基准
    hnsw_recall = np.mean([calculate_recall(bf, hnsw) 
                          for bf, hnsw in zip(bf_results_list, hnsw_results_list)])
    ivf_recall = np.mean([calculate_recall(bf, ivf) 
                         for bf, ivf in zip(bf_results_list, ivf_results_list)])
    
    print(f"\n召回率对比 (top_k={top_k}):")
    print(f"  暴力检索 (Brute Force): {bf_recall:.4f} (基准)")
    print(f"  IVF索引:                 {ivf_recall:.4f}")
    print(f"  HNSW索引:                {hnsw_recall:.4f}")
    
    # 性能对比
    print(f"\n性能对比:")
    start = time.time()
    for query in queries:
        db.brute_force_search(query, top_k=10)
    bf_time = (time.time() - start) / n_queries * 1000
    
    start = time.time()
    for query in queries:
        db.ivf_search(query, top_k=10)
    ivf_time = (time.time() - start) / n_queries * 1000
    
    start = time.time()
    for query in queries:
        db.hnsw_search(query, top_k=10)
    hnsw_time = (time.time() - start) / n_queries * 1000
    
    print(f"  暴力检索: {bf_time:.3f}ms/查询")
    print(f"  IVF索引:  {ivf_time:.3f}ms/查询")
    print(f"  HNSW索引: {hnsw_time:.3f}ms/查询")
    print(f"\n  IVF加速比: {bf_time/ivf_time:.1f}x")
    print(f"  HNSW加速比: {bf_time/hnsw_time:.1f}x")
    
    db.save()


def demo_comparison():
    """对比不同检索方法"""
    print("\n" + "=" * 60)
    print("对比不同检索方法")
    print("=" * 60)
    
    import time
    
    db = VerVectorDB(vector_dim=128, db_path="demo_compare.pkl")
    
    n_vectors = 5000
    vectors = np.random.random((n_vectors, 128)).astype(np.float32)
    db.batch_insert(vectors)
    print(f"插入{n_vectors}个向量")
    
    db.build_ivf_index(n_clusters=16)
    db.build_hnsw_index()
    print("索引构建完成")
    
    query = np.random.random(128).astype(np.float32)
    
    start = time.time()
    bf_results = db.brute_force_search(query, top_k=10)
    bf_time = time.time() - start
    print(f"暴力检索耗时: {bf_time*1000:.2f}ms")
    
    start = time.time()
    ivf_results = db.ivf_search(query, top_k=10)
    ivf_time = time.time() - start
    print(f"IVF检索耗时: {ivf_time*1000:.2f}ms")
    
    start = time.time()
    hnsw_results = db.hnsw_search(query, top_k=10)
    hnsw_time = time.time() - start
    print(f"HNSW检索耗时: {hnsw_time*1000:.2f}ms")
    
    print(f"\n暴力检索结果数: {len(bf_results)}")
    print(f"IVF检索结果数: {len(ivf_results)}")
    print(f"HNSW检索结果数: {len(hnsw_results)}")
    
    db.save()


if __name__ == "__main__":
    # 运行所有演示
    demo_hnsw_index()
    demo_pq_compression()
    demo_batch_operations()
    demo_filtered_search()
    demo_recall_comparison()  # 新增召回率对比
    demo_comparison()
    
    print("\n" + "=" * 60)
    print("所有演示完成!")
    print("=" * 60)
