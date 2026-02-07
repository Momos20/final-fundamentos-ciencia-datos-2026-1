# src/groq_page.py
import json
import os
import requests
import streamlit as st

DEFAULT_MODEL = "llama-3.1-8b-instant"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Parámetros fijos (no editables por el usuario)
FIXED_TEMPERATURE = 1.0
FIXED_MAX_TOKENS = 1800


def get_groq_key() -> str:
    """
    Obtiene la API Key de forma segura:
    1) Streamlit Secrets (Community Cloud)
    2) Variable de entorno
    3) Retorna "" si no existe
    """
    try:
        if "GROQ_API_KEY" in st.secrets:
            key = st.secrets["GROQ_API_KEY"]
            return key.strip() if isinstance(key, str) else ""
    except Exception:
        pass

    key = os.getenv("GROQ_API_KEY", "")
    return key.strip() if isinstance(key, str) else ""


def groq_chat_completion(api_key: str, model: str, messages: list):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": FIXED_TEMPERATURE,
        "max_tokens": FIXED_MAX_TOKENS,
    }
    r = requests.post(
        GROQ_API_URL,
        headers=headers,
        data=json.dumps(payload),
        timeout=60,
    )
    r.raise_for_status()
    return r.json()


def build_dataset_brief(bundle: dict, kpi_pack: dict) -> str:
    data_raw = bundle["data_raw"]
    data_imp = bundle["data_imp"]

    lines = []
    lines.append("Contexto del análisis (Saber Pro):")
    lines.append(f"- Filas: {data_raw.shape[0]}, columnas: {data_raw.shape[1]}")
    lines.append(f"- Nulos promedio (%): {data_raw.isna().mean().mean()*100:.2f}")
    lines.append(f"- Kruskal (N_NULOS vs target) p-value: {bundle.get('kw_p')}")
    lines.append(f"- Dataset procesado: filas {data_imp.shape[0]}, columnas {data_imp.shape[1]}")

    if kpi_pack.get("p_nat") is not None:
        lines.append(f"- P(nacional) bajo: {kpi_pack['p_nat']*100:.2f}%")
    if kpi_pack.get("KRAS") is not None:
        lines.append(f"- KRAS (pp): {kpi_pack['KRAS']*100:.2f}")
    if kpi_pack.get("FD") is not None:
        lines.append(f"- FD (pp): {kpi_pack['FD']*100:.2f}")
    if kpi_pack.get("KCC") is not None:
        lines.append(f"- KCC (pp): {kpi_pack['KCC']*100:.2f}")
    if kpi_pack.get("KDS") is not None:
        lines.append(f"- KDS: {kpi_pack['KDS']:.4f}")

    return "\n".join(lines)


def render_groq_page(bundle: dict, kpi_pack: dict):
    st.title("Groq IA")
    st.caption(
        "Asistente para explicar hallazgos, redactar insights de política pública y sugerir análisis adicionales."
    )

    # -------------------------
    # Sidebar
    # -------------------------
    with st.sidebar:
        st.subheader("Configuración Groq")

        api_key = get_groq_key()

        if not api_key:
            api_key = st.text_input(
                "Groq API Key",
                type="password",
                help="Clave usada solo en esta sesión si no existe en Secrets o variables de entorno.",
            )

        model = st.selectbox(
            "Modelo",
            options=[
                "llama-3.1-8b-instant",
                "llama-3.3-70b-versatile",
            ],
            index=0,
        )

        include_context = st.checkbox(
            "Incluir contexto del dataset (recomendado)",
            value=True,
        )

    # -------------------------
    # Estado del chat
    # -------------------------
    if "groq_chat" not in st.session_state:
        st.session_state.groq_chat = []

    st.subheader("Chat")
    for msg in st.session_state.groq_chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_msg = st.chat_input(
        "Escriba una pregunta (ej: 'Redacte 5 insights ejecutivos para MinEducación')"
    )

    if user_msg:
        if not api_key:
            st.error("No se encontró una Groq API Key válida.")
            return

        system = (
            "Usted es un analista de datos senior. Responda en español, claro y accionable. "
            "Evite causalidad (solo asociaciones) y respalde con evidencia cuantitativa."
        )

        messages = [{"role": "system", "content": system}]

        if include_context:
            messages.append(
                {
                    "role": "system",
                    "content": build_dataset_brief(bundle, kpi_pack),
                }
            )

        history = st.session_state.groq_chat[-10:]
        messages.extend(history)
        messages.append({"role": "user", "content": user_msg})

        st.session_state.groq_chat.append({"role": "user", "content": user_msg})
        with st.chat_message("user"):
            st.markdown(user_msg)

        try:
            resp = groq_chat_completion(
                api_key=api_key,
                model=model,
                messages=messages,
            )
            assistant_text = resp["choices"][0]["message"]["content"]

            st.session_state.groq_chat.append(
                {"role": "assistant", "content": assistant_text}
            )
            with st.chat_message("assistant"):
                st.markdown(assistant_text)

        except requests.HTTPError as e:
            st.error(
                f"Error HTTP Groq: {e}\n\nDetalle: {getattr(e.response, 'text', '')}"
            )
        except Exception as e:
            st.error(f"Error: {e}")

    # -------------------------
    # Acciones rápidas
    # -------------------------
    st.markdown("---")
    st.subheader("Acciones rápidas (prompts)")
    c1, c2, c3 = st.columns(3)

    with c1:
        if st.button("Insights ejecutivos"):
            st.session_state.groq_chat.append(
                {
                    "role": "user",
                    "content": (
                        "Redacte 7 insights ejecutivos accionables (no causales) "
                        "para MinEducación basados en KRAS/FD/KCC/KDS y territorio."
                    ),
                }
            )
            st.rerun()

    with c2:
        if st.button("Recomendaciones de política"):
            st.session_state.groq_chat.append(
                {
                    "role": "user",
                    "content": (
                        "Proponga 5 líneas de intervención (conectividad, tutorías, "
                        "focalización territorial) priorizadas por impacto y equidad. "
                        "Justifique con números."
                    ),
                }
            )
            st.rerun()

    with c3:
        if st.button("Próximos análisis"):
            st.session_state.groq_chat.append(
                {
                    "role": "user",
                    "content": (
                        "Sugiera próximos análisis estadísticos/ML (sin implementarlos) "
                        "que complementen este diagnóstico, indicando variables y métricas."
                    ),
                }
            )
            st.rerun()
