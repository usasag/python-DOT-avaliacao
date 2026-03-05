import faiss
import json
import sys
import numpy as np
from sentence_transformers import SentenceTransformer


def carregar_arquivos_locais() -> tuple[faiss.IndexFlatL2, list[dict]]:
    try:
        print("Lendo índice FAISS...")
        indice = faiss.read_index("faiss_index.bin")
        
        print("Carregando documentos...")
        with open("documentos.json", "r", encoding="utf-8") as f:
            documentos = json.load(f)
            
        return indice, documentos
    except FileNotFoundError:
        print("\n[ERRO] Arquivos de índice não encontrados. Execute gerador_indice.py primeiro.\n")
        sys.exit(1)


def buscar_documentos_relevantes(
    query: str, 
    modelo: SentenceTransformer, 
    indice: faiss.IndexFlatL2, 
    documentos: list[dict], 
    top_k: int = 5
) -> list[dict]:
    query_vector = modelo.encode([query], convert_to_numpy=True).astype(np.float32)
    distancias, indices_retornados = indice.search(query_vector, top_k)
    
    resultados = []
    for i, idx in enumerate(indices_retornados[0]):
        if idx != -1:
            doc_original = documentos[idx].copy()
            doc_original["distanciaL2"] = float(distancias[0][i])
            resultados.append(doc_original)
            
    return resultados


import re

def colorir(texto: str, cor: str = "93") -> str:
    """Retorna o texto envolto no código ANSI da cor escolhida (padrão 93: amarelo brilhante)."""
    return f"\033[{cor}m{texto}\033[0m"

def extrair_e_destacar(texto: str, query: str, limite: int = 400) -> str:
    partes = texto.split(" - ", 1)
    if len(partes) > 1 and len(partes[0]) < 100:
        titulo = partes[0].strip()
        corpo = partes[1].strip()
    else:
        titulo = "Sem título"
        corpo = texto.strip()

    query_terms = [t for t in re.split(r'\W+', query.lower()) if len(t) > 3]
    corpo_lower = corpo.lower()
    
    idx_central = -1
    for t in query_terms:
        idx = corpo_lower.find(t)
        if idx != -1:
            idx_central = idx
            break

    res_str = f"  Título: \033[1m{titulo}\033[0m\n"

    if idx_central == -1:
        trecho = corpo[:limite]
        if len(corpo) > limite:
            trecho += "..."
        res_str += f"  Trecho (Busca por Similaridade):\n    {trecho}"
    else:
        start = max(0, idx_central - 80)
        end = min(len(corpo), idx_central + limite)
        trecho = corpo[start:end]
        
        prefix = "..." if start > 0 else ""
        suffix = "..." if end < len(corpo) else ""
        
        for t in query_terms:
            pattern = re.compile(re.escape(t), re.IGNORECASE)
            trecho = pattern.sub(lambda m: colorir(m.group(0), "93"), trecho)
            
        res_str += f"  Trecho:\n    {prefix}{trecho}{suffix}"
        
    return res_str


def main():
    print("="*60)
    print("SERVIÇO DE BUSCA VETORIAL")
    print("="*60)
    
    indice_faiss, documentos = carregar_arquivos_locais()
    
    print("Carregando o modelo de linguagem ('all-MiniLM-L6-v2')...")
    modelo = SentenceTransformer("all-MiniLM-L6-v2")
    
    print(f"\nTotal: {indice_faiss.ntotal} documentos indexados.")
    print("="*60)
    print("Sistema pronto. Digite 'sair' para encerrar.")
    print("="*60)
    
    while True:
        try:
            query = input("\nPesquisar por: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nEncerrando.")
            sys.exit(0)
            
        if not query:
            continue
            
        if query.lower() in ("sair", "exit"):
            print("\nEncerrando.")
            break
            
        print("Buscando...", end="\r", flush=True)
        resultados = buscar_documentos_relevantes(query, modelo, indice_faiss, documentos, top_k=5)
        print(" " * 50, end="\r", flush=True)
        
        for k, res in enumerate(resultados):
            print(f"\n[Top {k+1}] ID: {res['id']} | Distância (L2): {res['distanciaL2']:.4f}")
            print(extrair_e_destacar(res['texto'], query, limite=400))


if __name__ == "__main__":
    main()
