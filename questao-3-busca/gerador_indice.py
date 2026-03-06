"""
gerador_indice.py — Geração e persistência do índice FAISS.

Decisão Arquitetural:
    O sistema de busca é dividido em duas fases: Ingestão (este script) e
    Consulta (`buscar.py`). Este script baixa um dataset real via HuggingFace,
    gera embeddings em lote com `sentence-transformers` e persiste o índice
    FAISS em disco, desacoplando o custo de vetorização do tempo de busca.
"""

import json
import time
import faiss
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer


def carregar_dados_ag_news(num_amostras: int = 1500) -> list[dict]:
    """
    Baixa e filtra notícias de tecnologia do dataset ag_news (HuggingFace).

    Filtra apenas a categoria Sci/Tech (label == 3) e retorna até
    `num_amostras` documentos formatados como dicionários {id, texto}.
    """
    print(f"Buscando dataset no HuggingFace (limite: {num_amostras} amostras)...")
    dataset = load_dataset("ag_news", split="train")

    # label 3 = categoria Sci/Tech
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
    """
    Gera embeddings para os documentos e persiste o índice FAISS em disco.

    Utiliza o modelo SentenceTransformer para codificar os textos em vetores
    de dimensão fixa (384 para all-MiniLM-L6-v2), constrói um índice FAISS
    IndexFlatL2 (busca exata por distância euclidiana) e salva os artefatos:
      - faiss_index.bin: índice vetorial binário
      - documentos.json: metadados dos documentos (id + texto original)
    """
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
    print("=" * 60) # Só pra ficar mais visível
    print("GERAÇÃO DE ÍNDICE DE BUSCA")
    print("=" * 60) # Só pra ficar mais visível

    docs = carregar_dados_ag_news(num_amostras=1500)
    gerar_vetores_e_salvar(docs)

    print("=" * 60) # Só pra ficar mais visível
    print("Processo concluído com sucesso.")
