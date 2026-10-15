# 🥷 giugtsu da kethy (ia de transformar rosto com seedance)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1JGfoDO8qtCQ8UYMaib9k1fImOX3Q6d-1)

Pipeline híbrido de **Motion Transfer** e substituição realista de personagens e identidades em vídeo, reproduzindo a capacidade do **Higgsfield Genjutsu (Reality Manipulation)** combinando o melhor do ecossistema Open Source gratuito com a API do **ByteDance Seedance 2.5**.

---

## ⚡ Como Funciona a Esteira

Em vez de pagar assinaturas caras por segundos de vídeo, o **giugtsu da kethy** divide a computação em duas etapas:

1. **Pré-Processamento Open Source Gratuito (Google Colab / GPU T4 ou Local):**
   * **DWPose (ControlNet)**: Rastreia a biomecânica e o esqueleto 3D da coreografia original frame a frame.
   * **SAM 2 (Meta AI)**: Segmentação e recorte contínuo dos corpos sem distorcer o cenário de fundo.
   * **InsightFace (Soul ID)**: Extrai os vetores biométricos das fotos do novo personagem em múltiplos ângulos, travando o rosto contra deformações.
2. **Renderização Neural na Nuvem (API Seedance 2.5):**
   * Envia apenas os parâmetros limpos (esqueleto + máscaras + fotos de referência) para o modelo de difusão da ByteDance sintetizar o vídeo final fotorrealista.

---

## 🚀 Como Usar no Google Colab (Recomendado)

1. Clique no badge acima **[Open In Colab](https://colab.research.google.com/drive/1JGfoDO8qtCQ8UYMaib9k1fImOX3Q6d-1)**.
2. Certifique-se de que o ambiente de execução está com a GPU ativa (`Ambiente de Execução > Alterar tipo de ambiente de execução > T4 GPU`).
3. Execute as células em sequência:
   * **Célula 1 a 3**: Instalação automática das bibliotecas e pesos neurais.
   * **Célula 4**: Faça o upload do vídeo da dança (`.mp4`) e das fotos do personagem (`.jpg`/`.png`).
   * **Célula 5 e 6**: Extração automática do esqueleto e da biometria.
   * **Célula 7**: Insira a sua chave da API do Seedance 2.5 e dispare a renderização do vídeo final.

---

## 💻 Como Rodar Localmente / Servidor Próprio

### 1. Clonar o repositório:
```bash
git clone https://github.com/kethysb/giugtsu-da-kethy-ia-de-transformar-rosto-com-seedance.git
cd giugtsu-da-kethy-ia-de-transformar-rosto-com-seedance
```

### 2. Instalar as dependências:
```bash
pip install -r requirements.txt
```

### 3. Executar o pipeline:
```bash
python pipeline.py --video inputs/danca.mp4 --images inputs/ --prompt "Cirilo dancing Brazilian funk, 8k, photorealistic"
```

---

## 📁 Estrutura do Repositório

```text
├── motion_transfer_seedance_pipeline.ipynb   # Notebook oficial do Google Colab
├── pipeline.py                               # Script CLI executável
├── requirements.txt                          # Dependências Python
├── README.md                                 # Documentação oficial
└── .gitignore                                # Exclusão de arquivos pesados e mídias
```

Desenvolvido para criar vídeos virais e transferências de movimento com custo mínimo e máxima fidelidade visual.
