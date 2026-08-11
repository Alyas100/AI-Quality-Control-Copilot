"""
Module 3: Agentic AI Copilot (LLM Integration)
================================================================================
Takes the outputs of Module 1 (Vision Grader) and Module 2 (Predictive Engine),
packages them into a structured JSON payload, and sends that payload to an LLM
acting as a Palm Oil Mill Operations Expert. Returns a concise, plain-language
action plan.

Supports both Anthropic Claude and OpenAI as providers. If no API key is
supplied, or if the live call fails for any reason (bad key, network issue,
rate limit -- all common at a hackathon on conference wifi), this module
transparently falls back to a rule-based recommendation engine so the demo
never simply breaks.
"""

import json
from typing import Optional, Tuple
import os
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()

SYSTEM_PROMPT = """You are a senior Palm Oil Mill Operations Expert with 20+ years of experience in Crude Palm Oil (CPO) quality control and Free Fatty Acid (FFA) management.

You will be given structured sensor and vision-analysis data (JSON) for one Fresh Fruit Bunch (FFB) batch. Produce a concise, plain-language operational action plan for the mill supervisor on shift.

Your response MUST:
1. Open with a risk classification in bold: **CRITICAL RISK**, **WARNING**, or **SAFE**.
2. In 2-4 sentences, name the specific contributing factors (ripeness, harvest delay, storage temperature, humidity) actually driving this batch's predicted FFA -- don't list all four generically, cite the ones that matter here.
3. End with one concrete, actionable recommendation that includes a specific timeframe (e.g. "fast-track to sterilizer within 2 hours").

Keep the entire response under 80 words. Write it as flowing operational guidance a mill supervisor could act on immediately -- no markdown headers, no bullet lists, no restating the raw numbers back verbatim."""

FALLBACK_NOTE = "\n\n_(Offline rule-based estimate -- live AI service unavailable: {error})_"


def build_payload(vision_result: dict, env_conditions: dict, predicted_ffa: float, risk_level: dict) -> dict:
    """Assemble the structured JSON payload passed to the LLM (and shown to the user)."""
    return {
        "vision_analysis": {
            "ripeness_category": vision_result["category"],
            "confidence": vision_result["confidence"],
            "ripeness_score": vision_result["ripeness_score"],
        },
        "environmental_conditions": {
            "harvest_delay_hours": round(float(env_conditions["harvest_delay_hours"]), 1),
            "storage_temp_c": round(float(env_conditions["storage_temp_c"]), 1),
            "humidity_percent": round(float(env_conditions["humidity_percent"]), 1),
        },
        "predictive_analysis": {
            "predicted_ffa_percentage": round(float(predicted_ffa), 2),
            "risk_level": risk_level["level"],
        },
        "thresholds": {"safe_max": 2.5, "warning_max": 3.5},
    }


def generate_fallback_recommendation(payload: dict) -> str:
    """Deterministic, rule-based stand-in for the LLM. Used when no API key is
    set or the live call fails, so the copilot always returns *something*."""
    ffa = payload["predictive_analysis"]["predicted_ffa_percentage"]
    risk = payload["predictive_analysis"]["risk_level"]
    ripeness = payload["vision_analysis"]["ripeness_category"]
    delay = payload["environmental_conditions"]["harvest_delay_hours"]
    temp = payload["environmental_conditions"]["storage_temp_c"]
    humidity = payload["environmental_conditions"]["humidity_percent"]

    factors = []
    if delay > 24:
        factors.append(f"a {delay:.0f}-hour harvest delay")
    if temp > 35:
        factors.append(f"elevated storage temperature ({temp:.0f}°C)")
    if humidity > 85:
        factors.append(f"high humidity ({humidity:.0f}%)")
    if ripeness in ("Overripe", "Rotted"):
        factors.append(f"{ripeness.lower()} fruit condition")
    factor_text = ", ".join(factors) if factors else "generally acceptable handling conditions"

    if risk == "Critical":
        action = "Fast-track this batch to the sterilizer within 2 hours and flag it for priority processing to limit further FFA rise."
    elif risk == "Warning":
        action = "Schedule processing within the next 6-8 hours and monitor storage conditions closely to prevent escalation."
    else:
        action = "Standard processing schedule is acceptable; continue routine monitoring."

    return f"**{risk.upper()} RISK**: Predicted FFA of {ffa:.2f}% is driven by {factor_text}. {action}"


def _call_gemini(payload: dict, api_key: str, model: str) -> str:
    import google.generativeai as genai
    
    genai.configure(api_key=api_key)
    
    # Gemini handles system prompts during model initialization
    generative_model = genai.GenerativeModel(
        model_name=model,
        system_instruction=SYSTEM_PROMPT
    )
    
    user_prompt = f"Batch analysis data:\n{json.dumps(payload, indent=2)}\n\nProvide the operational action plan."
    
    response = generative_model.generate_content(user_prompt)
    return response.text.strip()


def generate_action_plan(
    payload: dict, provider: str, api_key: Optional[str], model: str
) -> Tuple[str, bool]:
    """Generate the Copilot's action plan.

    Returns (message, used_live_llm). Falls back to a rule-based recommendation
    (used_live_llm=False) if no key is provided or the live call raises.
    """
    # If the UI doesn't provide a key, try to pull it from the .env file
    if not api_key:
        api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return generate_fallback_recommendation(payload), False

    try:
        provider == "gemini"
        text = _call_gemini(payload, api_key, model)
      

        if not text:
            raise ValueError("Empty response from LLM")
        return text, True

    except Exception as e:
        fallback = generate_fallback_recommendation(payload)
        return fallback + FALLBACK_NOTE.format(error=str(e)[:140]), False