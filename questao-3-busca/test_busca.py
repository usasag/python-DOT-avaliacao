"""
test_busca.py — Testes unitários para o sistema de busca semântica.

Utiliza um índice FAISS pequeno (in-memory) com documentos sintéticos
para validar o pipeline de busca sem depender do dataset real.
"""

import numpy as np
import pytest
import faiss
from sentence_transformers import SentenceTransformer

from buscar import buscar_documentos_relevantes, extrair_e_destacar, colorir


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def modelo():
    """Carrega o modelo SentenceTransformer uma vez para todos os testes."""
    return SentenceTransformer("all-MiniLM-L6-v2")


@pytest.fixture(scope="module")
def documentos_sinteticos():
    """Base de documentos sintéticos para testes."""
    return [
        {"id": 0, "texto": "NASA launches new Mars rover - The space agency sent a new rover to explore the red planet surface."},
        {"id": 1, "texto": "Python 3.12 released - New version of the programming language includes performance improvements."},
        {"id": 2, "texto": "Tesla unveils electric truck - The automaker revealed its new commercial electric vehicle."},
        {"id": 3, "texto": "OpenAI announces GPT-5 - The artificial intelligence lab introduces its latest language model."},
        {"id": 4, "texto": "Amazon expands cloud services - AWS adds new regions and machine learning features."},
        {"id": 5, "texto": "Apple introduces new MacBook - The laptop features the M4 chip and improved battery life."},
        {"id": 6, "texto": "Google develops quantum computer - The tech giant claims quantum supremacy breakthrough."},
        {"id": 7, "texto": "Microsoft acquires game studio - The acquisition strengthens Xbox exclusive game lineup."},
        {"id": 8, "texto": "SpaceX Starship test flight - The rocket completed its orbital test successfully."},
        {"id": 9, "texto": "Linux kernel 7.0 released - Major update brings improved hardware support and security."},
    ]


@pytest.fixture(scope="module")
def indice_faiss(modelo, documentos_sinteticos):
    """Cria um índice FAISS in-memory a partir dos documentos sintéticos."""
    textos = [doc["texto"] for doc in documentos_sinteticos]
    embeddings = modelo.encode(textos, convert_to_numpy=True).astype(np.float32)

    dimensao = embeddings.shape[1]
    indice = faiss.IndexFlatL2(dimensao)
    indice.add(embeddings)

    return indice


# ===========================================================================
#                        TESTES DE BUSCA
# ===========================================================================

class TestBuscaDocumentos:
    """Testes para a função buscar_documentos_relevantes."""

    def test_retorna_resultados(self, modelo, indice_faiss, documentos_sinteticos):
        """Verifica que a busca retorna resultados não-vazios."""
        resultados = buscar_documentos_relevantes(
            "space exploration", modelo, indice_faiss, documentos_sinteticos, top_k=3
        )
        assert len(resultados) > 0

    def test_respeita_top_k(self, modelo, indice_faiss, documentos_sinteticos):
        """Verifica que a busca retorna no máximo top_k resultados."""
        for k in [1, 3, 5]:
            resultados = buscar_documentos_relevantes(
                "programming language", modelo, indice_faiss, documentos_sinteticos, top_k=k
            )
            assert len(resultados) <= k

    def test_resultados_contem_distancia(self, modelo, indice_faiss, documentos_sinteticos):
        """Verifica que cada resultado contém a distância L2."""
        resultados = buscar_documentos_relevantes(
            "cloud computing", modelo, indice_faiss, documentos_sinteticos, top_k=3
        )
        for res in resultados:
            assert "distanciaL2" in res
            assert isinstance(res["distanciaL2"], float)
            assert res["distanciaL2"] >= 0

    def test_resultados_ordenados_por_distancia(self, modelo, indice_faiss, documentos_sinteticos):
        """Verifica que os resultados estão ordenados por distância crescente."""
        resultados = buscar_documentos_relevantes(
            "artificial intelligence", modelo, indice_faiss, documentos_sinteticos, top_k=5
        )
        distancias = [r["distanciaL2"] for r in resultados]
        assert distancias == sorted(distancias)

    def test_relevancia_semantica(self, modelo, indice_faiss, documentos_sinteticos):
        """Verifica que a busca por 'Mars rover' retorna o documento de NASA como Top 1."""
        resultados = buscar_documentos_relevantes(
            "Mars rover", modelo, indice_faiss, documentos_sinteticos, top_k=1
        )
        assert resultados[0]["id"] == 0  # NASA document

    def test_relevancia_semantica_programacao(self, modelo, indice_faiss, documentos_sinteticos):
        """Verifica que a busca por 'programming' retorna Python ou Linux primeiro."""
        resultados = buscar_documentos_relevantes(
            "programming language update", modelo, indice_faiss, documentos_sinteticos, top_k=2
        )
        ids_retornados = [r["id"] for r in resultados]
        # Deve retornar Python (id=1) e/ou Linux (id=9)
        assert 1 in ids_retornados or 9 in ids_retornados

    def test_preserva_campos_originais(self, modelo, indice_faiss, documentos_sinteticos):
        """Verifica que os campos id e texto originais são preservados."""
        resultados = buscar_documentos_relevantes(
            "electric vehicle", modelo, indice_faiss, documentos_sinteticos, top_k=1
        )
        assert "id" in resultados[0]
        assert "texto" in resultados[0]
        assert isinstance(resultados[0]["texto"], str)


# ===========================================================================
#                        TESTES DE FORMATAÇÃO
# ===========================================================================

class TestFormatacao:
    """Testes para as funções de formatação de saída."""

    def test_colorir_aplica_ansi(self):
        """Verifica que a função colorir envolve o texto em códigos ANSI."""
        resultado = colorir("teste", "93")
        assert "\033[93m" in resultado
        assert "teste" in resultado
        assert "\033[0m" in resultado

    def test_extrair_titulo(self):
        """Verifica que o título é extraído corretamente do formato ag_news."""
        texto = "Título do Artigo - Corpo do texto com informações relevantes."
        resultado = extrair_e_destacar(texto, "informações")
        assert "Título do Artigo" in resultado

    def test_extrair_sem_titulo(self):
        """Verifica o fallback 'Sem título' quando o formato não segue o padrão."""
        texto = "Texto sem separador de título com informações técnicas."
        resultado = extrair_e_destacar(texto, "xyz_sem_match")
        assert "Sem título" in resultado

    def test_destaque_termos_encontrados(self):
        """Verifica que termos encontrados recebem destaque ANSI."""
        texto = "Título - Este texto contém Python como linguagem principal"
        resultado = extrair_e_destacar(texto, "Python language")
        # O termo "Python" deve ser destacado (envolvido em códigos ANSI)
        assert "\033[93m" in resultado

    def test_trecho_similaridade_quando_sem_match(self):
        """Verifica que exibe 'Busca por Similaridade' quando não há match textual."""
        texto = "Título - Corpo do texto sobre assuntos diversos."
        resultado = extrair_e_destacar(texto, "xyzw_inexistente_1234")
        assert "Similaridade" in resultado
