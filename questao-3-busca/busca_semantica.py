import os
import time
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from datasets import load_dataset


def carregar_dados_mock(num_amostras: int = 300) -> list[dict]:
    """
    Carrega uma amostra maior de dados usando a biblioteca HuggingFace Datasets.
    
    Para simular "posts de blog sobre IA e tecnologia", vamos baixar uma pequena
    amostra do dataset 'ag_news' (notícias) e filtrar pela categoria Sci/Tech.
    
    Args:
        num_amostras: Quantidade de documentos para retornar.
        
    Returns:
        Uma lista de dicionários contendo 'id' e 'texto'.
    """
    print(f"Buscando dataset no HuggingFace (limitando a {num_amostras} amostras Sci/Tech)...")
    
    # ag_news possui labels: 0=World, 1=Sports, 2=Business, 3=Sci/Tech
    # Baixamos via streaming ou load direto (é um dataset pequeno)
    dataset = load_dataset("ag_news", split="train")
    
    # Filtra apenas Sci/Tech (label == 3)
    dataset_tech = dataset.filter(lambda example: example["label"] == 3)
    
    # Pegamos as 'num_amostras' primeiras
    amostras = dataset_tech.select(range(min(num_amostras, len(dataset_tech))))
    
    documentos = []
    for i, item in enumerate(amostras):
        documentos.append({
            "id": i,
            "titulo": f"Tech News #{i}", 
            "texto": item["text"]
        })
        
    print(f"{len(documentos)} documentos carregados com sucesso!\n")
    return documentos


def criar_indice_faiss(documentos: list[dict], modelo: SentenceTransformer) -> faiss.IndexFlatL2:
    """
    Converte os textos em embeddings e os armazena em um índice FAISS.
    
    ===========================================================================
    Matemática Abstrata do FAISS e Embeddings:
    ===========================================================================
    1. O `SentenceTransformer` atua como uma função `f: Texto -> R^N`. 
       Ele converte blocos de texto contínuos em um vetor denso no espaço 
       N-Dimensional (para o modelo all-MiniLM-L6-v2, N = 384).
       Nesse espaço contínuo, textos com significados semânticos parecidos
       (ex: "Inteligência Artificial" e "Machine Learning") ficam posicionados
       como pontos matematicamente próximos.

    2. O `FAISS` (Facebook AI Similarity Search) é uma biblioteca de 
       armazenamento vetorizado e busca de similaridade extremamente otimizada.
       
    3. Aqui utilizamos o `IndexFlatL2`. "L2" significa Distância Euclidiana.
       A distância Euclidiana d(p, q) entre dois pontos p e q em P^N é a 
       raíz quadrada da soma dos quadrados das diferenças de suas coordenadas:
       
           L2 = sqrt( sum_{i=1}^{N} (p_i - q_i)^2 )
           
       No `IndexFlatL2`, a busca é "Exhaustive" (força bruta): ele compara
       o vetor da query com *todos* os vetores salvos e retorna os K vetores
       com o menor valor de L2 (menor distância = maior similaridade).
    ===========================================================================
    
    Args:
        documentos: Lista de dicionários contendo a chave 'texto'.
        modelo: Instância do SentenceTransformer para vetorização.
        
    Returns:
        Um objeto FAISS index populado com os embeddings.
    """
    print("Vetorizando os documentos (gerando Embeddings no espaço R^384)...")
    textos = [doc["texto"] for doc in documentos]
    
    # As embeddings retornam como uma matriz Numpy de shape (num_docs, dimensao)
    start_time = time.time()
    embeddings = modelo.encode(textos, convert_to_numpy=True)
    embeddings = embeddings.astype(np.float32)  # FAISS exige float32
    print(f"Embeddings geradas em {time.time() - start_time:.2f} segundos.")
    
    dimensao = embeddings.shape[1] # Deve ser 384 para o all-MiniLM-L6-v2
    
    print(f"Inicializando banco FAISS (IndexFlatL2) para vetores de {dimensao} dimensões...")
    indice = faiss.IndexFlatL2(dimensao)
    
    # Adicionando os vetores ao índice. Em índices Flat, o FAISS apenas 
    # copia os vetores para uma matriz contígua em C++ otimizada.
    indice.add(embeddings)
    
    print(f"Total de vetores no índice FAISS: {indice.ntotal}\n")
    return indice


def buscar_documentos_relevantes(
    query: str, 
    modelo: SentenceTransformer, 
    indice: faiss.IndexFlatL2, 
    documentos: list[dict], 
    top_k: int = 2
) -> list[dict]:
    """
    Recebe a pergunta do usuário e busca no FAISS os dados mais similares semanticamente.
    
    Args:
        query: O texto ou pergunta buscado.
        modelo: O modelo SentenceTransformer (para gerar o vetor bidirecional da query).
        indice: O banco de dados FAISS populado.
        documentos: A lista original de documentos (para puxar o texto correspondente).
        top_k: Quantidade de documentos a serem retornados.
        
    Returns:
        Lista de dicionários contendo os documentos encontrados e sua "distância L2".
    """
    # 1. Converte a string da query para as coordenadas matemáticas (embedding)
    query_vector = modelo.encode([query], convert_to_numpy=True)
    query_vector = query_vector.astype(np.float32)
    
    # 2. Busca no índice FAISS. 
    # Retorna as Distâncias L2 (scores) e os IDs (índices posicinais)
    distancias, indices_retornados = indice.search(query_vector, top_k)
    
    resultados = []
    
    # As matrizes de distâncias e índices retornados têm shape (1, top_k) 
    # porque passamos apenas 1 query.
    for i, idx in enumerate(indices_retornados[0]):
        # idx pode ser -1 se não houver vetores suficientes no índice (não aplicável aqui)
        if idx != -1:
            doc_original = documentos[idx].copy()
            # Distância Euclidean L2 (quanto menor, mais similar semanticamente é o texto)
            distancia_l2 = distancias[0][i] 
            doc_original["distanciaL2"] = float(distancia_l2)
            resultados.append(doc_original)
            
    return resultados


def main():
    print("="*60)
    print("🧠 Sistema de Busca Semântica (FAISS + SentenceTransformers)")
    print("="*60)
    
    # 1. Carregar Dataset (Mock de dezenas/centenas de artigos)
    documentos = carregar_dados_mock(num_amostras=200)
    
    # 2. Inicializar o Modelo de Emdeddings
    # 'all-MiniLM-L6-v2' é extremamente pequeno, rápido e com altíssima performance 
    # para tarefas em inglês (e performance razoável em outras linguagens).
    # Como as notícias do `ag_news` estão em inglês, ele é ideal e gratuito.
    print("Carregando o modelo de linguagem SentenceTransformer ('all-MiniLM-L6-v2')...")
    modelo = SentenceTransformer("all-MiniLM-L6-v2")
    
    # 3. Gerar Embeddings e Inicializar FAISS (Vector DB)
    indice_faiss = criar_indice_faiss(documentos, modelo)
    
    # -------------------------------------------------------------------------
    # 4. Demonstração de Consultas
    # -------------------------------------------------------------------------
    consultas_exemplo = [
        "Which company is developing autonomous driving and intelligent cars?",
        "New advancements in open source software and programming"
    ]
    
    print("="*60)
    print("🚀 Demonstrando Buscas Semânticas")
    print("="*60)
    
    for query in consultas_exemplo:
        print(f"\n🔍 Query do Usuário: '{query}'")
        resultados = buscar_documentos_relevantes(query, modelo, indice_faiss, documentos, top_k=2)
        
        for k, res in enumerate(resultados):
            print(f"  [Top {k+1}] Document ID: {res['id']} | Distância Euclidiana L2 = {res['distanciaL2']:.4f}")
            print(f"  Texto: {res['texto'][:150]}...") # Exibindo os 150 primeiros caracteres
            print(f"  ---")


if __name__ == "__main__":
    main()
