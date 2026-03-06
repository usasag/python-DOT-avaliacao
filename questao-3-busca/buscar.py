"""
buscar.py — Motor de Busca Semântica Interativa (FAISS).

Decisão Arquitetural:
    Este script atua na fase de Inferência (Query). Carrega o índice FAISS e os
    metadados persistidos em disco pelo `gerador_indice.py`, evitando reprocessar
    embeddings. O SentenceTransformer é instanciado apenas para vetorizar a string
    de busca do usuário (operação O(1)), permitindo busca euclidiana (L2)
    ultrarrápida entre os vetores cacheados.
"""

import json
import re
import sys

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def carregar_arquivos_locais() -> tuple[faiss.IndexFlatL2, list[dict]]:
    """Carrega o índice FAISS e os metadados de documentos persistidos em disco."""
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
    """
    Busca os documentos mais similares à query usando distância L2.

    Gera o embedding da query e compara com os vetores armazenados no
    índice FAISS, retornando os `top_k` documentos mais próximos.
    """
    query_vector = modelo.encode([query], convert_to_numpy=True).astype(np.float32)
    distancias, indices_retornados = indice.search(query_vector, top_k)

    resultados = []
    for i, idx in enumerate(indices_retornados[0]):
        if idx != -1:
            doc_original = documentos[idx].copy()
            doc_original["distanciaL2"] = float(distancias[0][i])
            resultados.append(doc_original)

    return resultados


def colorir(texto: str, cor: str = "93") -> str:
    """Retorna o texto envolto no código ANSI da cor escolhida (padrão: amarelo)."""
    return f"\033[{cor}m{texto}\033[0m"


def extrair_e_destacar(texto: str, query: str, limite: int = 400) -> str:
    """
    Extrai título e trecho relevante do documento, destacando termos da busca.

    Se encontra um termo da query no corpo do texto, centraliza o trecho em
    torno dele e aplica destaque ANSI. Caso contrário, exibe os primeiros
    caracteres como trecho genérico de similaridade.
    """
    # Separa título do corpo (formato ag_news: "Título - Corpo do texto")
    partes = texto.split(" - ", 1)
    if len(partes) > 1 and len(partes[0]) < 100:
        titulo = partes[0].strip()
        corpo = partes[1].strip()
    else:
        titulo = "Sem título"
        corpo = texto.strip()

    # Filtra termos curtos da query (<=3 chars) para evitar falso positivos
    query_terms = [t for t in re.split(r'\W+', query.lower()) if len(t) > 3]
    corpo_lower = corpo.lower()

    # Busca a posição do primeiro termo encontrado no corpo
    idx_central = -1
    for t in query_terms:
        idx = corpo_lower.find(t)
        if idx != -1:
            idx_central = idx
            break

    res_str = f"  Título: \033[1m{titulo}\033[0m\n"

    if idx_central == -1:
        # Nenhum termo direto encontrado — match puramente semântico
        trecho = corpo[:limite]
        if len(corpo) > limite:
            trecho += "..."
        res_str += f"  Trecho (Busca por Similaridade):\n    {trecho}"
    else:
        # Centraliza o trecho em torno do termo encontrado
        start = max(0, idx_central - 80)
        end = min(len(corpo), idx_central + limite)
        trecho = corpo[start:end]

        prefix = "..." if start > 0 else ""
        suffix = "..." if end < len(corpo) else ""

        # Aplica destaque amarelo nos termos da query
        for t in query_terms:
            pattern = re.compile(re.escape(t), re.IGNORECASE)
            trecho = pattern.sub(lambda m: colorir(m.group(0), "93"), trecho)

        res_str += f"  Trecho:\n    {prefix}{trecho}{suffix}"

    return res_str


def main():
    """Ponto de entrada: carrega recursos e inicia loop interativo de busca."""
    print("=" * 60)
    print("SERVIÇO DE BUSCA VETORIAL")
    print("=" * 60)

    indice_faiss, documentos = carregar_arquivos_locais()

    print("Carregando o modelo de linguagem ('all-MiniLM-L6-v2')...")
    modelo = SentenceTransformer("all-MiniLM-L6-v2")

    print(f"\nTotal: {indice_faiss.ntotal} documentos indexados.")
    print("=" * 60)
    print("Sistema pronto. Digite 'sair' para encerrar.")
    print("=" * 60)

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
