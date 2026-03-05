# Questão 3 — Busca Semântica Vetorial (FAISS + SentenceTransformers)

Sistema local de busca semântica para IA e Programação, desenvolvido do zero utilizando embeddings via HuggingFace e indexação vetorial com FAISS.

A arquitetura foi dividida em dois ciclos de execução independentes simulando um banco de dados real (Batch Indexing + Live Inference).

---

## 🛠️ Tecnologias

| Tecnologia | Finalidade |
|---|---|
| **Python 3.10+** | Linguagem principal |
| **faiss-cpu** | Banco de dados vetorial eficiente para cálculos de similaridade (criado pelo Facebook AI) |
| **sentence-transformers** | Geração local de embeddings (converte texto em vetores matemáticos) |
| **datasets** | Biblioteca do HuggingFace para carregamento rápido de amostras de dados reais (mock em larga escala) |
| **numpy** | Manipulação rápida das matrizes geradas pelo Transformer na qual o FAISS opera nativamente |

---

## 📁 Estrutura do Projeto

```
questao-3-busca/
├── gerador_indice.py    # (Job 1) Ingestão. Baixa dados, cria vetores, salva no disco.
├── buscar.py            # (Job 2) Inferência. Lê do disco, recebe as queries e busca.
├── requirements.txt     # Pacotes da solução
└── README.md
```

---

## 🚀 Como Executar

### 1. Acessar o ambiente

```bash
cd questao-3-busca
python -m venv venv

# Windows
.\venv\Scripts\activate
# Linux/macOS
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Gerar Banco de Dados Vetorial (Pre-compute)

O primeiro passo arquiteta o *Vector Store*. O script baixará 1.500 notícias de tecnologia (usando a base publicamente disponível `ag_news`), as transformará em **Embeddings (R^384)** e fará o salvamento diretamente em sua máquina (persistência).

```bash
python gerador_indice.py
```
*(Gera os artefatos físicos `faiss_index.bin` e `documentos.json` na sua pasta).*

### 3. Executar o Motor de Busca (Iterativo)

Depois que a indexação pesada for despachada para o disco físico e não viver apenas em RAM, você pode ligar e desligar o sistema de busca livremente de maneira ultrarrápida.

```bash
python buscar.py
```

O console se abrirá recebendo infinitas queries interativas de busca e apresentando os Top 5 resultados ordenados pelas **Menores Distâncias L2**.

---

## 📐 Lógica Matemática e Arquitetura

1. `gerador_indice.py` instancia `SentenceTransformers` atuando matematicamente como uma função $f: Texto \rightarrow \mathbb{R}^{384}$. Transformamos cada frase em coordenadas de 384 dimensões em Batch O(N).
2. O `FAISS` usa o índice `faiss.IndexFlatL2` que repousa as matrizes contíguas de Numpy perfeitamente em memória C++ e serializa tudo isso com perfeição via `.bin`.
3. Em `buscar.py` fazemos apenas a desserialização e o Embedding On-the-Fly de *uma frase alvo*. O algoritmo FAISS de forca-bruta iterage a matriz alvo usando a **Distância Euclidiana ($L^2$)** contra as metades já serializadas na etapa 2. Menor distância significa Maior Igualdade Semântica.
