import faiss
import json
import time
import numpy as np
from sentence_transformers import SentenceTransformer
from datasets import load_dataset


def carregar_dados_ag_news(num_amostras: int = 1500) -> list[dict]:
    print(f"Buscando dataset no HuggingFace (limite: {num_amostras} amostras)...")
    dataset = load_dataset("ag_news", split="train")
    dataset_tech = dataset.filter(lambda example: example["label"] == 3)
    
    amostras = dataset_tech.select(range(min(num_amostras, len(dataset_tech))))
    
    documentos = []
    for i, item in enumerate(amostras):
        documentos.append({
            "id": i,
            "texto": item["text"]
        })
        
    print(f"{len(documentos)} documentos carregados com sucesso.\n")
    return documentos


def gerar_vetores_e_salvar(documentos: list[dict], modelo_nome: str = "all-MiniLM-L6-v2"):
    print(f"Carregando o modelo SentenceTransformer ('{modelo_nome}')...")
    modelo = SentenceTransformer(modelo_nome)
    
    print("Vetorizando os documentos...")
    textos = [doc["texto"] for doc in documentos]
    
    start_time = time.time()
    embeddings = modelo.encode(textos, convert_to_numpy=True)
    embeddings = embeddings.astype(np.float32)
    print(f"Embeddings geradas em {time.time() - start_time:.2f} segundos.")
    
    dimensao = embeddings.shape[1] 
    
    print(f"Inicializando índice FAISS e populando {len(embeddings)} vetores...")
    indice = faiss.IndexFlatL2(dimensao)
    indice.add(embeddings)
    
    faiss.write_index(indice, "faiss_index.bin")
    print("Índice salvo como 'faiss_index.bin'.")
    
    with open("documentos.json", "w", encoding="utf-8") as f:
        json.dump(documentos, f, ensure_ascii=False, indent=2)
    print("Metadados salvos como 'documentos.json'.")


if __name__ == "__main__":
    print("="*60)
    print("GERAÇÃO DE ÍNDICE DE BUSCA")
    print("="*60)
    
    docs = carregar_dados_ag_news(num_amostras=1500)
    gerar_vetores_e_salvar(docs)
    
    print("="*60)
    print("Processo concluído com sucesso.")
