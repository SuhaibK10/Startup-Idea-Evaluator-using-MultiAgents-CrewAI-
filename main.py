import os
from dotenv import load_dotenv
import streamlit as st
from langchain_openai import ChatOpenAI
from crewai import Agent, Task

load_dotenv()

# LLM via OpenRouter OpenAI-compatible , Since OpenAI API is chargeable 
def make_llm():

    return ChatOpenAI(
        model=os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
        base_url=os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1"),
        api_key=os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY"),
        default_headers={
            k: v for k, v in {
                "HTTP-Referer": os.getenv("HTTP_REFERER"),
                "X-Title": os.getenv("X_TITLE", "StartupIdeaEvaluator")
            }.items() if v
        }
    )

def textify(x) -> str:
    """Return plain string from CrewAI TaskOutput or any object."""
    
    if hasattr(x, "raw") and isinstance(getattr(x, "raw"), str):
        return x.raw

    for attr in ("output", "final_output", "text", "content"):
        if hasattr(x, attr) and isinstance(getattr(x, attr), str):
            return getattr(x, attr)
    
    if isinstance(x, (list, tuple)):
        return "\n\n".join(textify(i) for i in x)
    return str(x) if x is not None else ""

def join_ctx(*parts) -> str:
    return "\n\n---\n\n".join(textify(p) for p in parts if p)

#  Streamlit for UI 
st.set_page_config(page_title="Startup Idea Evaluator", page_icon="🚀", layout="centered")
st.title("🚀 Startup Idea Evaluator (CrewAI)")

with st.form("idea_form"):
    idea = st.text_area(
        "Describe your startup idea",
        height=120,
        placeholder="e.g., AI voice bot for clinics to book appointments and answer FAQs"
    )
    target = st.text_input("Target customer", placeholder="e.g., dermatology clinics in Delhi NCR")
    region = st.text_input("Market/Region", placeholder="e.g., India")
    price = st.text_input("Intended pricing", placeholder="e.g., ₹2,999 per month")
    submitted = st.form_submit_button("Run Evaluation")

if submitted:
    if not idea.strip():
        st.warning("Please enter your startup idea.")
        st.stop()

    llm = make_llm()
    problem_ctx = f"Idea: {idea}\nTarget: {target or 'N/A'}\nRegion: {region or 'N/A'}\nPricing: {price or 'N/A'}"

    # Agent- Validator 
    validator = Agent(
        name="Problem Validator",
        role="Validates problem-solution fit",
        goal="Decide if the problem is real and painful enough to solve now.",
        backstory="PM who interviews users and kills weak ideas early.",
        llm=llm, verbose=True
    )
    #Agent- Researcher
    researcher = Agent(
        name="Market Researcher",
        role="Quant & qual market research",
        goal="Size the market, map competitors, and find insights.",
        backstory="Analyst triangulating public data and reasonable assumptions.",
        llm=llm, verbose=True
    )
    #Agent - Modeler
    modeler = Agent(
        name="Business Model Builder",
        role="Designs business model & GTM",
        goal="Propose pricing, costs, GTM, and a simple unit economics check.",
        backstory="Operator thinking in CAC, LTV, margins, and payback.",
        llm=llm, verbose=True
    )
    #Agent- Risker
    risker = Agent(
        name="Risk Analyzer",
        role="Identifies risks and mitigations",
        goal="Surface product, market, execution, legal, and moat risks with mitigations.",
        backstory="Skeptical advisor who stress-tests assumptions.",
        llm=llm, verbose=True
    )

    # Tasks 
    t_validate = Task(
        description=(
            "Evaluate the PROBLEM and its urgency.\n"
            f"{problem_ctx}\n\n"
            "Output:\n"
            "- Problem statement in one line\n"
            "- Who is suffering & how often\n"
            "- Current workarounds\n"
           
            "- Verdict: green/yellow/red with 2–3 lines why"
        ),
        agent=validator,
        expected_output="Concise bullets + a Green/Yellow/Red verdict with justification."
    )

    t_research = Task(
        description=(
            "Market & competition.\n"
            f"{problem_ctx}\n\n"
            "Output:\n"
            "- TAM/SAM/SOM (rough ranges) with assumptions\n"
            "- 3–5 competitors (direct/indirect) and quick notes\n"
            "- Differentiators\n"
            "- Early adopter segment\n"
            "- 3 insights you’d bet on"
        ),
        agent=researcher,
        expected_output="Numbers + bullets, compact."
    )

    t_bizmodel = Task(
        description=(
            "Business model & GTM design.\n"
            f"{problem_ctx}\n\n"
            "Use insights from prior tasks. Output:\n"
            "- Pricing model (Free/Pro/Enterprise) with example tiers\n"
            "- Simple unit economics: ARPU, gross margin guess, CAC guess, payback\n"
            "- Top 3 channels and a 30-day GTM plan\n"
            "- 3 traction KPIs for first 60–90 days"
        ),
        agent=modeler,
        expected_output="Clear plan + one small unit-econ math example."
    )

    t_risks = Task(
        description=(
            "Risk assessment & mitigations.\n"
            f"{problem_ctx}\n\n"
            "Output:\n"
            "- Product risks\n"
            "- Market/Timing risks\n"
            "- Execution risks\n"
            "- Legal/Compliance risks\n"
            "- Moat & defensibility\n"
            "- Mitigations (bulleted, concrete)"
        ),
        agent=risker,
        expected_output="Bulleted risks grouped by type + concrete mitigations."
    )


    with st.spinner("Running Problem Validator…"):
        out_validate_obj = t_validate.execute_sync(agent=validator, context=None, tools=[])
        out_validate = textify(out_validate_obj)
    with st.expander("🧩 Problem Validator Output", expanded=True):
        st.markdown(out_validate)

    with st.spinner("Running Market Researcher…"):
        out_research_obj = t_research.execute_sync(agent=researcher, context=out_validate, tools=[])
        out_research = textify(out_research_obj)
    with st.expander("📊 Market Researcher Output", expanded=True):
        st.markdown(out_research)

    with st.spinner("Running Business Model Builder…"):
        ctx_bm = join_ctx(out_validate, out_research)
        out_bizmodel_obj = t_bizmodel.execute_sync(agent=modeler, context=ctx_bm, tools=[])
        out_bizmodel = textify(out_bizmodel_obj)
    with st.expander("💼 Business Model Builder Output", expanded=True):
        st.markdown(out_bizmodel)

    with st.spinner("Running Risk Analyzer…"):
        ctx_risk = join_ctx(out_validate, out_research, out_bizmodel)
        out_risks_obj = t_risks.execute_sync(agent=risker, context=ctx_risk, tools=[])
        out_risks = textify(out_risks_obj)
    with st.expander("⚠️ Risk Analyzer Output", expanded=True):
        st.markdown(out_risks)

    st.success("✅ Evaluation complete. Expand the sections above to view each agent’s output.")
