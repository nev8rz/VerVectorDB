import numpy as np
import pickle
import uuid
import heapq
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity


class HNSWNode:
    """HNSW节点"""
    def __init__(self, vector_id, vector):
        self.vector_id = vector_id
        self.vector = vector
        self.neighbors = {}


class HNSWIndex:
    """HNSW索引实现 - 优化版"""
    def __init__(self, M=32, ef_construction=200, mL=1/np.log(2)):
        self.M = M
        self.ef_construction = ef_construction
        self.mL = mL
        
        self.nodes = {}
        self.entry_point = None
        self.max_level = 0
        
    def _get_random_level(self):
        return int(-np.log(np.random.random()) * self.mL)
    
    def _similarity(self, v1, v2):
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return np.dot(v1, v2) / (norm1 * norm2)
    
    def search_layer(self, query_vector, ef, level):
        visited = set()
        candidates = []  # 最大堆 [(-similarity, node_id)]
        result = []      # 最小堆 [(similarity, node_id)]
        
        if self.entry_point is None:
            return []
        
        ep = self.entry_point
        ep_sim = self._similarity(query_vector, self.nodes[ep].vector)
        heapq.heappush(candidates, (-ep_sim, ep))
        heapq.heappush(result, (ep_sim, ep))
        visited.add(ep)
        
        while candidates:
            neg_sim, node_id = heapq.heappop(candidates)
            sim = -neg_sim
            
            if result:
                worst_sim, _ = result[0]
            
            if result and sim <= worst_sim:
                break
            
            current_node = self.nodes[node_id]
            
            if level < len(current_node.neighbors):
                for neighbor_id, neg_s in current_node.neighbors[level]:
                    if neighbor_id not in visited:
                        visited.add(neighbor_id)
                        new_sim = self._similarity(query_vector, self.nodes[neighbor_id].vector)
                        
                        heapq.heappush(candidates, (-new_sim, neighbor_id))
                        heapq.heappush(result, (new_sim, neighbor_id))
                        
                        if len(result) > ef:
                            heapq.heappop(result)
        
        return [(nid, -sim) for sim, nid in result]
    
    def select_neighbors(self, query_vector, candidates, M):
        if not candidates:
            return []
        
        similarities = []
        for nid in candidates:
            if nid in self.nodes:
                sim = self._similarity(query_vector, self.nodes[nid].vector)
                similarities.append((sim, nid))
        
        similarities.sort(reverse=True)
        return [(nid, -sim) for sim, nid in similarities[:M]]
    
    def insert(self, vector_id, vector):
        if vector_id in self.nodes:
            return
            
        node = HNSWNode(vector_id, vector)
        self.nodes[vector_id] = node
        
        level = self._get_random_level()
        level = min(level, 5)
        
        if self.entry_point is None:
            self.entry_point = vector_id
            self.max_level = level
            node.neighbors = {i: [] for i in range(level + 1)}
            return
        
        if level > self.max_level:
            self.max_level = level
            for nid in self.nodes:
                while len(self.nodes[nid].neighbors) <= level:
                    max_lvl = len(self.nodes[nid].neighbors)
                    self.nodes[nid].neighbors[max_lvl] = []
        
        node.neighbors = {i: [] for i in range(level + 1)}
        
        ep = self.entry_point
        for lvl in range(self.max_level, -1, -1):
            if lvl > level:
                continue
            candidates = self.search_layer(vector, 1, lvl)
            if candidates:
                ep = candidates[0][0]
        
        for lvl in range(min(level, self.max_level) + 1):
            candidates = self.search_layer(vector, self.ef_construction, lvl)
            candidate_ids = [nid for nid, _ in candidates if nid != vector_id]
            
            neighbors = self.select_neighbors(vector, candidate_ids, self.M)
            
            node.neighbors[lvl] = neighbors
            for neighbor_id, neg_sim in neighbors:
                self.nodes[neighbor_id].neighbors[lvl].append((vector_id, neg_sim))
        
        if level == self.max_level and level > 0:
            self.entry_point = vector_id
    
    def search(self, query_vector, top_k=5, ef_search=200):
        if self.entry_point is None or not self.nodes:
            return []
        
        for lvl in range(self.max_level, 0, -1):
            candidates = self.search_layer(query_vector, 1, lvl)
            if candidates:
                pass
        
        results = self.search_layer(query_vector, max(ef_search, top_k * 2), 0)
        results.sort(key=lambda x: x[1])
        
        return results[:top_k]


class PQCompressor:
    """Product Quantization压缩器"""
    def __init__(self, n_subquantizers=8, n_bits=8):
        self.n_subquantizers = n_subquantizers
        self.n_bits = n_bits
        self.n_centroids = 2 ** n_bits
        self.codebooks = []
        self.is_fitted = False
        
    def fit(self, vectors):
        n, dim = vectors.shape
        if dim % self.n_subquantizers != 0:
            raise ValueError(f"向量维度{dim}必须能被n_subquantizers({self.n_subquantizers})整除")
        
        sub_dim = dim // self.n_subquantizers
        
        for i in range(self.n_subquantizers):
            start = i * sub_dim
            end = start + sub_dim
            sub_vectors = vectors[:, start:end]
            
            kmeans = KMeans(n_clusters=self.n_centroids, random_state=42, n_init=3)
            kmeans.fit(sub_vectors)
            self.codebooks.append(kmeans.cluster_centers_)
        
        self.is_fitted = True
        self.original_dim = dim
        self.sub_dim = sub_dim
        print(f"PQ模型训练完成: {self.n_subquantizers}个子空间, 每子空间{self.n_centroids}个质心")
    
    def encode(self, vectors):
        if not self.is_fitted:
            raise ValueError("PQ模型未训练，请先调用fit()")
        
        vectors = np.atleast_2d(vectors).astype(np.float32)
        n, dim = vectors.shape
        sub_dim = dim // self.n_subquantizers
        
        codes = np.zeros((n, self.n_subquantizers), dtype=np.uint8)
        
        for i in range(self.n_subquantizers):
            start = i * sub_dim
            end = start + sub_dim
            sub_vectors = vectors[:, start:end]
            
            codebook = self.codebooks[i]
            distances = np.linalg.norm(sub_vectors[:, np.newaxis] - codebook, axis=2)
            codes[:, i] = np.argmin(distances, axis=1)
        
        return codes
    
    def decode(self, codes):
        if not self.is_fitted:
            raise ValueError("PQ模型未训练，请先调用fit()")
        
        codes = np.atleast_2d(codes).astype(np.uint8)
        n = codes.shape[0]
        dim = self.n_subquantizers * self.sub_dim
        
        reconstructed = np.zeros((n, dim), dtype=np.float32)
        
        for i in range(self.n_subquantizers):
            start = i * self.sub_dim
            end = start + self.sub_dim
            reconstructed[:, start:end] = self.codebooks[i][codes[:, i]]
        
        return reconstructed


class VerVectorDB:
    def __init__(self, vector_dim, db_path="vector_db.pkl", 
                 hnsw_M=32, hnsw_ef_construction=200,
                 pq_n_subquantizers=8, pq_n_bits=8):
        self.vector_dim = vector_dim
        self.db_path = db_path
        
        self.vectors = np.array([])
        self.vector_ids = []
        self.id_to_index = dict()
        self.metadata = dict()
        
        self.hnsw_index = HNSWIndex(M=hnsw_M, ef_construction=hnsw_ef_construction)
        self.use_hnsw = False
        
        self.ivf_index = None
        self.ivf_kmeans = None
        
        self.pq_compressor = PQCompressor(n_subquantizers=pq_n_subquantizers, n_bits=pq_n_bits)
        self.use_pq = False
        self.pq_codes = None
        
    def _check_vector_dim(self, vector):
        if len(vector) != self.vector_dim:
            raise ValueError(f"向量维度错误，需为{self.vector_dim}维，当前为{len(vector)}维")
    
    def insert(self, vector, metadata=None):
        vector = np.array(vector, dtype=np.float32).flatten()
        self._check_vector_dim(vector)
        
        vector_id = str(uuid.uuid4())
        
        if len(self.vectors) == 0:
            self.vectors = np.expand_dims(vector, axis=0)
        else:
            self.vectors = np.vstack([self.vectors, vector])
        
        self.vector_ids.append(vector_id)
        self.id_to_index[vector_id] = len(self.vector_ids) - 1
        
        if metadata:
            self.metadata[vector_id] = metadata
        
        if self.use_hnsw:
            self.hnsw_index.insert(vector_id, vector)
        
        return vector_id
    
    def batch_insert(self, vectors, metadatas=None):
        vectors = np.array(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = np.expand_dims(vectors, axis=0)
        
        n = len(vectors)
        for v in vectors:
            self._check_vector_dim(v)
        
        vector_ids = [str(uuid.uuid4()) for _ in range(n)]
        old_count = len(self.vectors)
        
        if old_count == 0:
            self.vectors = vectors.copy()
        else:
            self.vectors = np.vstack([self.vectors, vectors])
        
        for i, vid in enumerate(vector_ids):
            self.id_to_index[vid] = old_count + i
        
        self.vector_ids.extend(vector_ids)
        
        if metadatas:
            for i, vid in enumerate(vector_ids):
                self.metadata[vid] = metadatas[i]
        
        if self.use_hnsw:
            for i, vid in enumerate(vector_ids):
                self.hnsw_index.insert(vid, vectors[i])
        
        return vector_ids
    
    def get_by_id(self, vector_id):
        if vector_id not in self.id_to_index:
            raise KeyError(f"未找到ID为{vector_id}的向量")
        index = self.id_to_index[vector_id]
        return {
            "vector_id": vector_id,
            "vector": self.vectors[index].tolist(),
            "metadata": self.metadata.get(vector_id, {})
        }
    
    def update(self, vector_id, new_vector=None, new_metadata=None):
        if vector_id not in self.id_to_index:
            raise KeyError(f"未找到ID为{vector_id}的向量")
        index = self.id_to_index[vector_id]
        
        if new_vector is not None:
            new_vector = np.array(new_vector, dtype=np.float32).flatten()
            self._check_vector_dim(new_vector)
            self.vectors[index] = new_vector
        
        if new_metadata is not None:
            self.metadata[vector_id] = new_metadata
    
    def delete(self, vector_id):
        if vector_id not in self.id_to_index:
            raise KeyError(f"未找到ID为{vector_id}的向量")
        index = self.id_to_index[vector_id]
        
        self.vectors = np.delete(self.vectors, index, axis=0)
        self.vector_ids.pop(index)
        del self.id_to_index[vector_id]
        if vector_id in self.metadata:
            del self.metadata[vector_id]
        
        self.id_to_index = {vid: idx for idx, vid in enumerate(self.vector_ids)}
    
    def brute_force_search(self, query_vector, top_k=5, filter_func=None):
        if len(self.vectors) == 0:
            return []
        
        query_vector = np.array(query_vector, dtype=np.float32).flatten()
        self._check_vector_dim(query_vector)
        
        similarities = cosine_similarity([query_vector], self.vectors)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k * 3]
        
        results = []
        for idx in top_indices:
            vector_id = self.vector_ids[idx]
            meta = self.metadata.get(vector_id, {})
            
            if filter_func and not filter_func(meta):
                continue
            
            results.append({
                "vector_id": vector_id,
                "similarity": float(similarities[idx]),
                "vector": self.vectors[idx].tolist(),
                "metadata": meta
            })
            
            if len(results) >= top_k:
                break
        
        return results
    
    def build_hnsw_index(self):
        if len(self.vectors) == 0:
            raise ValueError("数据库中无向量数据")
        
        self.hnsw_index = HNSWIndex(M=self.hnsw_index.M, ef_construction=self.hnsw_index.ef_construction)
        
        for i, vid in enumerate(self.vector_ids):
            self.hnsw_index.insert(vid, self.vectors[i])
        
        self.use_hnsw = True
        print(f"HNSW索引构建完成，共{len(self.vectors)}个节点")
    
    def hnsw_search(self, query_vector, top_k=5, ef_search=200, filter_func=None):
        if not self.use_hnsw or self.hnsw_index.entry_point is None:
            raise ValueError("请先调用build_hnsw_index()构建HNSW索引")
        
        query_vector = np.array(query_vector, dtype=np.float32).flatten()
        self._check_vector_dim(query_vector)
        
        results = self.hnsw_index.search(query_vector, top_k * 3, ef_search)
        
        filtered_results = []
        for vector_id, neg_sim in results:
            if vector_id not in self.id_to_index:
                continue
            index = self.id_to_index[vector_id]
            similarity = -neg_sim
            meta = self.metadata.get(vector_id, {})
            
            if filter_func and not filter_func(meta):
                continue
            
            filtered_results.append({
                "vector_id": vector_id,
                "similarity": similarity,
                "vector": self.vectors[index].tolist(),
                "metadata": meta
            })
            
            if len(filtered_results) >= top_k:
                break
        
        return filtered_results
    
    def build_ivf_index(self, n_clusters=16):
        """构建IVF索引"""
        if len(self.vectors) == 0:
            raise ValueError("数据库中无向量数据")
        
        vectors_for_kmeans = self.vectors.astype(np.float64)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(vectors_for_kmeans)
        
        self.ivf_index = {i: [] for i in range(n_clusters)}
        for idx, label in enumerate(cluster_labels):
            self.ivf_index[label].append(idx)
        
        self.ivf_kmeans = kmeans
    
    def ivf_search(self, query_vector, top_k=5, filter_func=None):
        """IVF检索 - 优化版：扩大搜索范围保证召回率"""
        if self.ivf_index is None:
            raise ValueError("请先调用build_ivf_index()构建IVF索引")
        
        query_vector = np.array(query_vector, dtype=np.float64).flatten()
        self._check_vector_dim(query_vector)
        
        n_clusters = len(self.ivf_index)
        cluster_centers = self.ivf_kmeans.cluster_centers_
        
        # 计算到所有簇中心的相似度
        cluster_sims = cosine_similarity([query_vector], cluster_centers)[0]
        
        # 搜索一半的簇（保证召回率）
        n_search = max(n_clusters // 2, 8)
        top_clusters = np.argsort(cluster_sims)[::-1][:n_search]
        
        # 收集候选
        all_indices = []
        for cluster_id in top_clusters:
            all_indices.extend(self.ivf_index.get(cluster_id, []))
        
        if not all_indices:
            return []
        
        # 批量计算相似度
        candidate_vectors = self.vectors[all_indices]
        sims = cosine_similarity([query_vector], candidate_vectors)[0]
        
        # 构建候选列表
        all_candidates = []
        for idx, (orig_idx, sim) in enumerate(zip(all_indices, sims)):
            all_candidates.append({
                'original_idx': orig_idx,
                'vector_id': self.vector_ids[orig_idx],
                'similarity': float(sim),
                'cluster_id': int(top_clusters[0])
            })
        
        # 按相似度排序
        all_candidates.sort(key=lambda x: x['similarity'], reverse=True)
        
        results = []
        for cand in all_candidates[:top_k * 3]:
            meta = self.metadata.get(cand['vector_id'], {})
            
            if filter_func and not filter_func(meta):
                continue
            
            results.append({
                "vector_id": cand['vector_id'],
                "similarity": cand['similarity'],
                "vector": self.vectors[cand['original_idx']].tolist(),
                "metadata": meta,
                "cluster_id": cand['cluster_id']
            })
            
            if len(results) >= top_k:
                break
        
        return results
    
    def train_pq(self, sample_size=None):
        if len(self.vectors) == 0:
            raise ValueError("数据库中无向量数据")
        
        if sample_size is None:
            sample_size = min(10000, len(self.vectors))
        
        indices = np.random.choice(len(self.vectors), sample_size, replace=False)
        sample_vectors = self.vectors[indices]
        
        self.pq_compressor.fit(sample_vectors)
        self.use_pq = True
        print(f"PQ压缩模型训练完成，采样{sample_size}个向量")
    
    def compress(self):
        if not self.use_pq:
            raise ValueError("请先调用train_pq()训练PQ模型")
        
        self.pq_codes = self.pq_compressor.encode(self.vectors)
        original_size = self.vectors.nbytes
        compressed_size = self.pq_codes.nbytes
        print(f"向量压缩完成，存储空间减少约{1 - compressed_size / original_size:.2%}")
    
    def decompress(self, codes=None):
        if codes is None:
            codes = self.pq_codes
        return self.pq_compressor.decode(codes)
    
    def batch_search(self, query_vectors, top_k=5, method="brute_force", 
                     ef_search=100, filter_func=None, batch_size=32):
        query_vectors = np.array(query_vectors, dtype=np.float32)
        if query_vectors.ndim == 1:
            query_vectors = np.expand_dims(query_vectors, axis=0)
        
        results = []
        for i in range(0, len(query_vectors), batch_size):
            batch = query_vectors[i:i+batch_size]
            
            for query in batch:
                if method == "hnsw":
                    search_results = self.hnsw_search(query, top_k, ef_search, filter_func)
                elif method == "ivf":
                    search_results = self.ivf_search(query, top_k, filter_func)
                else:
                    search_results = self.brute_force_search(query, top_k, filter_func)
                results.append(search_results)
        
        return results
    
    def filtered_search(self, query_vector, top_k=5, 
                        keywords=None, metadata_filter=None, method="brute_force"):
        def filter_func(meta):
            if keywords:
                text = meta.get("text", "").lower()
                if not any(kw.lower() in text for kw in keywords):
                    return False
            if metadata_filter and not metadata_filter(meta):
                return False
            return True
        
        if method == "hnsw":
            return self.hnsw_search(query_vector, top_k, filter_func=filter_func)
        elif method == "ivf":
            return self.ivf_search(query_vector, top_k, filter_func=filter_func)
        else:
            return self.brute_force_search(query_vector, top_k, filter_func=filter_func)
    
    def save(self):
        data = {
            "vector_dim": self.vector_dim,
            "vectors": self.vectors,
            "vector_ids": self.vector_ids,
            "id_to_index": self.id_to_index,
            "metadata": self.metadata,
            "ivf_index": self.ivf_index,
            "ivf_kmeans": self.ivf_kmeans,
            "hnsw_index": self.hnsw_index,
            "use_hnsw": self.use_hnsw,
            "use_pq": self.use_pq,
            "pq_compressor": self.pq_compressor,
            "pq_codes": self.pq_codes
        }
        with open(self.db_path, "wb") as f:
            pickle.dump(data, f)
        print(f"数据库已保存至{self.db_path}")
    
    @classmethod
    def load(cls, db_path="vector_db.pkl"):
        with open(db_path, "rb") as f:
            data = pickle.load(f)
        
        db = cls(vector_dim=data["vector_dim"], db_path=db_path)
        db.vectors = data["vectors"]
        db.vector_ids = data["vector_ids"]
        db.id_to_index = data["id_to_index"]
        db.metadata = data["metadata"]
        db.ivf_index = data.get("ivf_index")
        db.ivf_kmeans = data.get("ivf_kmeans")
        db.hnsw_index = data.get("hnsw_index")
        db.use_hnsw = data.get("use_hnsw", False)
        db.use_pq = data.get("use_pq", False)
        db.pq_compressor = data.get("pq_compressor")
        db.pq_codes = data.get("pq_codes")
        
        print(f"已从{db_path}加载数据库，共包含{len(db.vectors)}个向量")
        return db
