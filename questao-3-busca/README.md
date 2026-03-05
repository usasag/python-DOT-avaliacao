# Questão 3 — Busca Semântica de Documentos (FAISS + SentenceTransformers)

Sistema local de busca semântica para IA e Programação, desenvolvido do zero utilizando embeddings via HuggingFace e indexação vetorial com FAISS.

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
├── busca_semantica.py   # Script principal comentado com lógica O(N) FAISS
├── requirements.txt     # Pacotes da solução
└── README.md
```

---

## 🚀 Como Executar

O dataset (ag_news Filtrado por Tecnologia) e o modelo (`all-MiniLM-L6-v2` ~90MB) são baixados e cacheados localmente na primeira execução.

### 1. Acessar o diretório

```bash
cd questao-3-busca
```

### 2. Criar e ativar o ambiente virtual

```bash
python -m venv venv

# Windows
.\venv\Scripts\activate
# Linux/macOS
source venv/bin/activate
```

### 3. Instalar dependências

A instalação pode demorar 1-2 minutos devida às dependências pesadas de IA (`torch`, `huggingface-hub`, etc) embutidas no pacote do sentence-transformer.

```bash
pip install -r requirements.txt
```

### 4. Executar o sistema de busca

```bash
python busca_semantica.py
```

O script carregará automaticamente 200 notícias de tecnologia, as transformará em Embeddings e fará consultas matemáticas comparatórias (Query -> Top 2 Resultados Mais Próximos).

---

## 📐 Lógica Matemática e Arquitetura

1. `SentenceTransformers` funciona como a função matemática $f: Texto \rightarrow \mathbb{R}^{384}$. Transformamos cada frase em coordenadas no "espaço semântico" de 384 dimensões.
2. O `FAISS` usa o índice `faiss.IndexFlatL2` que repousa as matrizes contíguas de Numpy perfeitamente em memória C++. 
3. Quando o usuário busca (uma Query), repetimos o Passo 1 apenas para a string da Pergunta (criando o vetor $\vec{q}$). Em seguida, o algoritmo FAISS itera com **Distância Euclidiana ($L^2$)** contra todo o banco para trazer rapidamente a menor distância (onde Menor Distância = Maior Semelhança Semântica).
