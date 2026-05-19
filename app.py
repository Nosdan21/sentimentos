"""
Análise de Sentimentos com Transformers — Streamlit App
Modelo: nlptown/bert-base-multilingual-uncased-sentiment
"""

import re
import io
from collections import defaultdict

import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from transformers import pipeline

# ── Configuração da página ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Análise de Sentimentos",
    page_icon="🧠",
    layout="wide",
)

# ── Constantes ────────────────────────────────────────────────────────────────
NOME_MODELO = "nlptown/bert-base-multilingual-uncased-sentiment"

MAPA_ESTRELAS = {
    1: "Muito Negativo 😠",
    2: "Negativo 😞",
    3: "Neutro / Misto 😐",
    4: "Positivo 🙂",
    5: "Muito Positivo 😄",
}

CORES = {
    1: "#d32f2f",
    2: "#f57c00",
    3: "#fbc02d",
    4: "#388e3c",
    5: "#1976d2",
}

FRASES_EXEMPLO = [
    "Estou muito feliz com o atendimento, foi excelente!",
    "O serviço foi horrível, estou muito insatisfeito.",
    "Não sei como me sinto sobre isso.",
    "Foi ok, mas poderia ser melhor.",
    "Péssima experiência, nunca mais compro aqui.",
    "Que móvel excelente! Melhor cadeira que já comprei.",
    "Tudo certo, conforme solicitado.",
]

# ── Funções auxiliares ────────────────────────────────────────────────────────

def extrair_estrelas(label: str) -> int:
    match = re.search(r"(\d)", label)
    return int(match.group(1)) if match else 3


def limpar_frase(texto: str) -> str:
    return texto.strip().rstrip(".")


@st.cache_resource(show_spinner="⏳ Carregando modelo BERT (~700 MB), aguarde...")
def carregar_modelo():
    return pipeline(
        "sentiment-analysis",
        model=NOME_MODELO,
        truncation=True,
        max_length=512,
    )


def analisar_frases(analyzer, frases: list[str]) -> list[dict]:
    resultados = []
    for frase in frases:
        if not frase.strip():
            continue
        try:
            r = analyzer(frase)[0]
            estrelas = extrair_estrelas(r["label"])
            resultados.append({
                "frase": frase,
                "estrelas": estrelas,
                "categoria": MAPA_ESTRELAS[estrelas],
                "confiança": round(r["score"] * 100, 2),
            })
        except Exception as e:
            resultados.append({
                "frase": frase,
                "estrelas": 0,
                "categoria": f"Erro: {e}",
                "confiança": 0.0,
            })
    return resultados


def gerar_graficos(resultados: list[dict]):
    contagem = defaultdict(int)
    for r in resultados:
        if r["estrelas"] > 0:
            contagem[r["estrelas"]] += 1

    estrelas_lista = list(range(1, 6))
    totais_lista = [contagem[e] for e in estrelas_lista]
    cores = [CORES[e] for e in estrelas_lista]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor("#0e1117")
    for ax in axes:
        ax.set_facecolor("#0e1117")
        ax.tick_params(colors="white")
        ax.title.set_color("white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

    # Barras
    bars = axes[0].bar(
        estrelas_lista, totais_lista, color=cores, edgecolor="#222", width=0.6
    )
    axes[0].set_xlabel("Estrelas")
    axes[0].set_ylabel("Quantidade de Frases")
    axes[0].set_title("Quantidade por Categoria")
    axes[0].set_xticks(estrelas_lista)
    axes[0].set_xticklabels([f"{e}⭐" for e in estrelas_lista], color="white")
    for bar, valor in zip(bars, totais_lista):
        if valor > 0:
            axes[0].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.05,
                str(valor),
                ha="center", va="bottom", fontweight="bold", color="white",
            )

    # Pizza
    dados_pizza = [(e, t) for e, t in zip(estrelas_lista, totais_lista) if t > 0]
    if dados_pizza:
        estrelas_p = [d[0] for d in dados_pizza]
        totais_p = [d[1] for d in dados_pizza]
        cores_p = [CORES[e] for e in estrelas_p]
        rotulos_p = [f"{e}★ ({t})" for e, t in dados_pizza]
        axes[1].pie(
            totais_p,
            labels=rotulos_p,
            colors=cores_p,
            autopct="%1.1f%%",
            startangle=90,
            pctdistance=0.75,
            textprops={"color": "white"},
        )
    axes[1].set_title("Proporção por Categoria")

    plt.tight_layout()
    return fig


# ── Interface ──────────────────────────────────────────────────────────────────

st.title("🧠 Análise de Sentimentos com Transformers")
st.markdown(
    "Modelo **`nlptown/bert-base-multilingual-uncased-sentiment`** — "
    "classifica textos de 1 a 5 estrelas, com suporte a português."
)

# Carrega o modelo uma única vez (cached)
analyzer = carregar_modelo()
st.success("✅ Modelo carregado e pronto para uso.")

st.divider()

# ── Abas ──────────────────────────────────────────────────────────────────────
aba_manual, aba_arquivo, aba_exemplo = st.tabs([
    "✍️  Digite frases",
    "📂  Upload de arquivo",
    "🔬  Frases de exemplo",
])

# ── Aba 1: Input manual ───────────────────────────────────────────────────────
with aba_manual:
    st.subheader("Digite uma frase por linha")
    texto_usuario = st.text_area(
        "Frases para analisar:",
        height=180,
        placeholder="Ex.:\nO produto chegou no prazo!\nAtendimento péssimo.",
    )
    if st.button("🔍 Analisar", key="btn_manual"):
        frases = [limpar_frase(l) for l in texto_usuario.splitlines() if l.strip()]
        if not frases:
            st.warning("Digite ao menos uma frase.")
        else:
            with st.spinner("Analisando..."):
                resultados = analisar_frases(analyzer, frases)

            # Resultados individuais
            st.subheader("Resultados")
            for r in resultados:
                stars = "⭐" * r["estrelas"] if r["estrelas"] > 0 else "⚠️"
                st.markdown(
                    f"**{r['frase']}**  \n"
                    f"{stars} &nbsp; {r['categoria']} &nbsp; "
                    f"*(confiança: {r['confiança']}%)*"
                )

            # Gráficos e tabela
            if len(resultados) > 1:
                st.divider()
                st.subheader("Distribuição")
                st.pyplot(gerar_graficos(resultados))

            st.divider()
            df = pd.DataFrame(resultados)
            st.dataframe(df, use_container_width=True)
            csv = df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
            st.download_button("⬇️ Baixar CSV", csv, "resultados.csv", "text/csv")


# ── Aba 2: Upload de arquivo ──────────────────────────────────────────────────
with aba_arquivo:
    st.subheader("Faça upload de um arquivo .txt (uma frase por linha)")
    arquivo = st.file_uploader("Selecione o arquivo", type=["txt"])

    if arquivo and st.button("🔍 Analisar arquivo", key="btn_arquivo"):
        conteudo = arquivo.read().decode("utf-8", errors="replace")
        frases = [limpar_frase(l) for l in conteudo.splitlines() if l.strip()]
        if not frases:
            st.warning("Arquivo vazio ou sem frases válidas.")
        else:
            st.info(f"✅ {len(frases)} frase(s) carregada(s).")
            progress = st.progress(0, text="Processando...")
            resultados = []
            for i, frase in enumerate(frases):
                resultados += analisar_frases(analyzer, [frase])
                progress.progress((i + 1) / len(frases), text=f"Processando {i+1}/{len(frases)}...")
            progress.empty()

            st.subheader("Resultados")
            for r in resultados:
                stars = "⭐" * r["estrelas"] if r["estrelas"] > 0 else "⚠️"
                st.markdown(
                    f"**{r['frase']}**  \n"
                    f"{stars} &nbsp; {r['categoria']} &nbsp; "
                    f"*(confiança: {r['confiança']}%)*"
                )

            st.divider()
            st.subheader("Distribuição")
            st.pyplot(gerar_graficos(resultados))

            df = pd.DataFrame(resultados)
            st.dataframe(df, use_container_width=True)
            csv = df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
            st.download_button("⬇️ Baixar CSV", csv, "resultados_arquivo.csv", "text/csv")


# ── Aba 3: Frases de exemplo ──────────────────────────────────────────────────
with aba_exemplo:
    st.subheader("Frases de teste embutidas")
    st.markdown("Clique no botão para rodar o conjunto de frases de sanity-check.")

    if st.button("🔬 Rodar exemplos", key="btn_exemplo"):
        with st.spinner("Analisando frases de exemplo..."):
            resultados = analisar_frases(analyzer, FRASES_EXEMPLO)

        for r in resultados:
            stars = "⭐" * r["estrelas"]
            col1, col2 = st.columns([3, 1])
            col1.markdown(f"**{r['frase']}**  \n{stars} &nbsp; {r['categoria']}")
            col2.metric("Confiança", f"{r['confiança']}%")

        st.divider()
        st.subheader("Distribuição")
        st.pyplot(gerar_graficos(resultados))

# ── Rodapé ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Sentimento 2025 · Modelo: nlptown/bert-base-multilingual-uncased-sentiment · "
    "Desenvolvido com 🤗 Transformers + Streamlit"
)
