#加载模型
import os
import jieba
from sentence_transformers import SentenceTransformer
from vervectordb import VerVectorDB
def read_txt_and_split(file_path):
    """读取txt文件内容并切分为句子列表"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"指定的txt文件不存在：{file_path}")
    # 读取文件（默认UTF-8编码，若有乱码可尝试gbk）
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    # 句子切分
    sentences = content.split("。")
    print(f"成功读取txt文件，共切分出 {len(sentences)} 个句子")
    return sentences

def text_to_vector(text, model):
    """将文本转换为向量（基于SentenceTransformer模型）"""
    # encode方法直接返回向量，convert_to_numpy=True确保输出为numpy数组
    return model.encode(text, convert_to_numpy=True)

if __name__ == "__main__":
    # 1. 配置参数（请根据实际情况修改txt文件路径和模型路径）
    txt_file = "./data/datawhale.txt"  # 你的txt文件路径
    model_dir = "../models/iic/nlp_gte_sentence-embedding_chinese-base"  # 预训练模型本地存储路径
    VECTOR_DIM = 768  # 主流中文语义模型输出维度多为768，可根据实际模型调整
    
    # 2. 加载中文语义向量模型（两种加载方式可选）
    print("正在加载中文文本嵌入模型...")
    model = SentenceTransformer(model_dir)
    print(f"成功加载本地模型：{model_dir}")

    # 3. 初始化向量数据库
    db = VerVectorDB(vector_dim=VECTOR_DIM)
    
    # 4. 读取txt文件并切分句子
    print(f"\n正在读取并处理txt文件：{txt_file}")
    sentences = read_txt_and_split(txt_file)
    if not sentences:
        raise ValueError("未从txt文件中提取到有效句子")
    
    # 5. 句子转向量并插入数据库（附带元数据：原始句子）
    print("\n正在将句子转向量并插入数据库...")
    embeddings = model.encode(sentences)  # 批量生成向量，效率更高
    for idx, (sentence, embedding) in enumerate(zip(sentences, embeddings), 1):
        # 元数据包含句子内容和序号，便于后续查看
        metadata = {"sentence": sentence, "sequence": idx}
        db.insert(embedding, metadata=metadata)
    
    # 6. 持久化数据库
    db.save()
    
    # 7. 测试暴力检索（查询与向量数据库相关的内容）
    print("\n=== 暴力检索结果（查询：'Datawhale有多个学习者参与活动'）===")
    query_text = "Datawhale有多个学习者参与活动"
    query_vector = text_to_vector(query_text, model)
    brute_results = db.brute_force_search(query_vector, top_k=3)
    for res in brute_results:
        print(f"相似度：{res['similarity']:.4f} | 句子：{res['metadata']['sentence']}")
    
    # 8. 构建IVF索引并测试近似检索
    print("\n=== IVF近似检索结果（查询：'Datawhale有多个学习者参与活动'）===")
    # 根据句子数量调整聚类数（一般为数据量的平方根左右）
    n_clusters = max(2, int(len(sentences)**0.5))
    db.build_ivf_index(n_clusters=n_clusters)
    ivf_results = db.ivf_search(query_vector, top_k=3)
    for res in ivf_results:
        print(f"相似度：{res['similarity']:.4f} | 聚类ID：{res['cluster_id']} | 句子：{res['metadata']['sentence']}")
    
    # 9. 测试数据更新与查询
    print("\n=== 数据更新与查询测试 ===")
    # 获取第一个向量的ID（即第一个句子对应的向量）
    first_vector_id = db.vector_ids[0]
    first_sentence = db.get_by_id(first_vector_id)['metadata']['sentence']
    print(f"待更新的原始句子：{first_sentence}")
    # 更新其元数据（模拟句子修正）
    new_metadata = {"sentence": f"【修正】{first_sentence}", "sequence": 1, "updated": True}
    db.update(first_vector_id, new_metadata=new_metadata)
    # 按ID查询更新结果
    updated_res = db.get_by_id(first_vector_id)
    print(f"更新后的数据：{updated_res['metadata']['sentence']}")
    
    # 10. 测试数据删除
    db.delete(first_vector_id)
    print(f"\n删除后数据库向量总数：{len(db.vectors)}")
    
    # 11. 从本地加载数据库验证持久化功能
    print("\n=== 从本地加载数据库 ===")
    loaded_db = VerVectorDB.load()
    print(f"加载的数据库向量总数：{len(loaded_db.vectors)}")
    # 验证加载的数据
    if loaded_db.vector_ids:
        sample_id = loaded_db.vector_ids[0]
        sample_data = loaded_db.get_by_id(sample_id)
        print(f"加载数据示例：{sample_data['metadata']['sentence']}")