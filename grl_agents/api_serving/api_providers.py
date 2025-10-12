import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    from litellm import completion
except Exception:  # pragma: no cover
    completion = None  # type: ignore


class LLMProviderError(Exception):
    pass


def chat_completion(
    messages: List[Dict[str, str]],
    provider: str = "openai",                 # "openai" or "gemini"
    model: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: Optional[int] = None,
    api_key: Optional[str] = None,
    extra_args: Optional[Dict[str, Any]] = None,
    *,
    route: str = "ai_studio",                 # "ai_studio" (default) or "vertex"
    api_version: str = "v1",                  # keep "v1" unless you need v1beta
) -> str:
    """
    Minimal LiteLLM wrapper with sane defaults:
      - Gemini via Google AI Studio (API key) by default, no ADC required
      - Versioned base URL to avoid 404s
      - Optional Vertex route (ADC or Vertex API key) when route="vertex"
    """
    if completion is None:
        raise LLMProviderError("litellm is not installed")

    provider_lower = provider.lower()
    if provider_lower not in {"openai", "gemini"}:
        raise LLMProviderError(f"Unsupported provider: {provider}")

    # If caller passed a key, set the right env var
    if api_key:
        if provider_lower == "openai":
            os.environ["OPENAI_API_KEY"] = api_key
        elif provider_lower == "gemini":
            if route == "vertex":
                # Vertex can use ADC or API key (Express). Keep caller in charge of auth.
                os.environ.setdefault("GOOGLE_API_KEY", api_key)
            else:
                # AI Studio route: use GEMINI_API_KEY
                os.environ["GEMINI_API_KEY"] = api_key

    # Default models
    if model is None:
        model = "gpt-4o-mini" if provider_lower == "openai" else "gemini/gemini-2.0-flash"

    kwargs: Dict[str, Any] = dict(model=model, messages=messages, temperature=temperature)
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    # Provider-specific defaults
    if provider_lower == "gemini":
        if not model.startswith("gemini/"):
            # LiteLLM uses gemini/<id> to select the AI Studio/Vertex provider path
            model_id = model
            model = f"gemini/{model_id}"
            kwargs["model"] = model

        if route == "vertex":
            # Use Vertex AI route (requires ADC or Vertex API key setup)
            kwargs.update({
                "custom_llm_provider": "vertex_ai_beta",
                # Let LiteLLM construct the Vertex URL; you just ensure ADC/keys are present.
                # Optionally pass vertex_project / vertex_location via extra_args if needed.
            })
        else:
            # Google AI Studio route (simple API key)
            # IMPORTANT: version the base to avoid 404s
            kwargs.update({
                "custom_llm_provider": "gemini",
                "api_base": f"https://generativelanguage.googleapis.com/{api_version}"
            })

    if extra_args:
        kwargs.update(extra_args)

    # Ensure LiteLLM receives Gemini API key explicitly (header X-Goog-Api-Key)
    if provider_lower == "gemini":
        if api_key:
            kwargs["api_key"] = api_key
        else:
            env_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if env_key:
                kwargs["api_key"] = env_key

    try:
        resp = completion(**kwargs)
        choice = resp.get("choices", [{}])[0]
        message = choice.get("message", {})

        # Normalize message into a dict-like structure
        def _get(obj, key, default=None):
            try:
                if isinstance(obj, dict):
                    return obj.get(key, default)
                return getattr(obj, key, default)
            except Exception:
                return default

        content = _get(message, "content")

        # If function/tool calls present, convert to our XML-like function blocks
        tool_calls_obj = _get(message, "tool_calls") or []
        if tool_calls_obj:
            blocks: List[str] = []
            for tc in tool_calls_obj:
                fn_obj = _get(tc, "function", {})
                name = _get(fn_obj, "name", "")
                args_raw = _get(fn_obj, "arguments", "{}")
                try:
                    args_dict = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
                except Exception:
                    args_dict = {"_raw": str(args_raw)}
                lines = [f"<function={name}>"]
                if isinstance(args_dict, dict):
                    for k, v in args_dict.items():
                        val = v if isinstance(v, str) else json.dumps(v)
                        lines.append(f"<parameter={k}>{val}</parameter>")
                lines.append("</function>")
                blocks.append("\n".join(lines))
            return "\n".join(blocks)

        # Otherwise, return plain content (may be empty string)
        return content or ""
    except Exception as e:
        raise LLMProviderError(str(e))


