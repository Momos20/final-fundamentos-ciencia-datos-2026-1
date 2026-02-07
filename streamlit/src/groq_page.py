# src/groq_page.py
import json
import os
import requests
import streamlit as st

DEFAULT_MODEL = "llama-3.1-8b-instant"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"


def get_groq_key() -> str:
    """
    Obtiene la API Key de forma segura:
    1) Streamlit Secrets (ideal para Streamlit Community Cloud)
    2) Variable de entorno (ideal para local/otros despliegues)
    3) Retorna "" si no existe
    """
    try:
        if "GROQ_API_KEY" in st.secrets:
            key = st.secrets["GROQ_API_KEY"]
            return key.strip() if isinstance(key, str) else ""
    except Exception:
        # Si st.secrets no está disponible por algún motivo, seguimos al env var
        pass

    key = os.getenv("GROQ_API_KEY", "")
    return key.strip() if isinstance(key, str) else ""


def groq_chat_completion(
    api_key: str,
    model: str,
    messages: list,
    temperature: float = 0.2,
    max_tokens: int = 1500,
):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    r = requests.post(GROQ_API_URL, headers=headers, data=json.dumps(payload), timeout=60)
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

    if "p_nat" in kpi_pack and kpi_pack["p_nat"] is not None:
        lines.append(f"- P(nacional) bajo: {kpi_pack['p_nat']*100:.2f}%")
    if "KRAS" in kpi_pack and kpi_pack["KRAS"] is not None:
        lines.append(f"- KRAS (pp): {kpi_pack['KRAS']*100:.2f}")
    if "FD" in kpi_pack and kpi_pack["FD"] is not None:
        lines.append(f"- FD (pp): {kpi_pack['FD']*100:.2f}")
    if "KCC" in kpi_pack and kpi_pack["KCC"] is not None:
        lines.append(f"- KCC (pp): {kpi_pack['KCC']*100:.2f}")
    if "KDS" in kpi_pack and kpi_pack["KDS"] is not None:
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

        # 1) Intentar cargar desde Secrets / env var
        api_key = get_groq_key()

        # 2) Si existe, no pedirla; si no, permitir input manual (fallback)
        if api_key:
            st.success("API Key cargada desde Secrets / entorno")
        else:
            api_key = st.text_input(
                "Groq API Key",
                type="password",
                help="No se encontró GROQ_API_KEY en Secrets ni en el entorno. Se usará solo en esta sesión.",
            )

        model = st.selectbox(
            "Modelo",
            options=[
                "llama-3.1-8b-instant",
                "llama-3.3-70b-versatile",
            ],
            index=0,
        )

        temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
        max_tokens = st.slider("Max tokens", 100, 1500, 600, 50)
        include_context = st.checkbox("Incluir contexto del dataset (recomendado)", value=True)

    # -------------------------
    # Chat state
    # -------------------------
    if "groq_chat" not in st.session_state:
        st.session_state.groq_chat = []

    st.subheader("Chat")
    for msg in st.session_state.groq_chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_msg = st.chat_input("Escriba una pregunta (ej: 'Redacte 5 insights ejecutivos para MinEducación')")

    if user_msg:
        # Validación final
        if not api_key:
            st.error("No se encontró una Groq API Key válida (Secrets/entorno/input).")
            return

        system = (
            "Usted es un analista de datos senior. Responda en español, claro y accionable. "
            "Cuando proponga hallazgos, evite causalidad (solo asociaciones) y sugiera evidencia cuantitativa."
        )

        messages = [{"role": "system", "content": system}]

        if include_context:
            brief = build_dataset_brief(bundle, kpi_pack)
            messages.append({"role": "system", "content": brief})

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
                temperature=temperature,
                max_tokens=max_tokens,
            )
            assistant_text = resp["choices"][0]["message"]["content"]

            st.session_state.groq_chat.append({"role": "assistant", "content": assistant_text})
            with st.chat_message("assistant"):
                st.markdown(assistant_text)

        except requests.HTTPError as e:
            st.error(f"Error HTTP Groq: {e}\n\nDetalle: {getattr(e.response, 'text', '')}")
        except Exception as e:
            st.error(f"Error: {e}")

    st.markdown("---")
    st.subheader("Acciones rápidas (prompts)")
    c1, c2, c3 = st.columns(3)

    with c1:
        if st.button("Insights ejecutivos"):
            st.session_state.groq_chat.append(
                {
                    "role": "user",
                    "content": "Redacte 7 insights ejecutivos accionables (no causales) para MinEducación basados en KRAS/FD/KCC/KDS y territorio.",
                }
            )
            st.rerun()

    with c2:
        if st.button("Recomendaciones de política"):
            st.session_state.groq_chat.append(
                {
                    "role": "user",
                    "content": "Proponga 5 líneas de intervención (conectividad, tutorías, focalización territorial) priorizadas por impacto y equidad. Justifique con números.",
                }
            )
            st.rerun()

    with c3:
        if st.button("Próximos análisis"):
            st.session_state.groq_chat.append(
                {
                    "role": "user",
                    "content": "Sugiera próximos análisis estadísticos/ML (sin implementarlos) que complementen este diagnóstico, indicando variables y métricas.",
                }
            )
            st.rerun()
