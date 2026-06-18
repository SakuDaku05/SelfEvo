"""
llm_protocol.py — Defines the minimal interface that any LLM client must
expose to work with SePO and the CorrectionAgent.

You do NOT need to subclass this; it is a structural (duck-typed) protocol.
Any object with a ``chat(messages)`` method works.

Adapters for common providers are included below so users can get started
immediately without writing boilerplate.

Quick start examples
--------------------

OpenAI::

    from evolution.llm_protocol import OpenAIAdapter
    llm = OpenAIAdapter(api_key="sk-…", model="gpt-4o")

Anthropic::

    from evolution.llm_protocol import AnthropicAdapter
    llm = AnthropicAdapter(api_key="sk-ant-…", model="claude-3-5-sonnet-20241022")

Google Gemini::

    from evolution.llm_protocol import GeminiAdapter
    llm = GeminiAdapter(api_key="AIza…", model="gemini-1.5-pro")

Ollama (local)::

    from evolution.llm_protocol import OllamaAdapter
    llm = OllamaAdapter(model="llama3")

Cohere::

    from evolution.llm_protocol import CohereAdapter
    llm = CohereAdapter(api_key="…", model="command-r-plus")

Custom (any provider)::

    class MyLLM:
        def chat(self, messages: list[dict]) -> str:
            # messages = [{"role": "user"|"assistant"|"system", "content": "…"}]
            response = my_sdk.call(messages)
            return response.text

    llm = MyLLM()

Pass ``llm`` into :class:`AgentConnector`::

    connector = AgentConnector(llm_client=llm)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


# ======================================================================= #
# Protocol definition                                                       #
# ======================================================================= #

@runtime_checkable
class LLMClientProtocol(Protocol):
    """Duck-typed protocol for LLM clients used by SePO and CorrectionAgent."""

    def chat(self, messages: List[Dict[str, str]]) -> str:
        """
        Send *messages* to the LLM and return the assistant reply as a string.

        Parameters
        ----------
        messages:
            OpenAI-style message list, e.g.
            ``[{"role": "system", "content": "…"}, {"role": "user", "content": "…"}]``

        Returns
        -------
        str
            The raw text of the model's reply.
        """
        ...


# ======================================================================= #
# Adapters                                                                  #
# ======================================================================= #

class OpenAIAdapter:
    """
    Adapter for the official ``openai`` Python SDK (v1+).

    Install: ``pip install openai``
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        base_url: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        try:
            import openai as _openai
        except ImportError as e:
            raise ImportError("openai package not installed. Run: pip install openai") from e
        self._client = _openai.OpenAI(api_key=api_key, base_url=base_url, **kwargs)
        self.model = model

    def chat(self, messages: List[Dict[str, str]]) -> str:
        resp = self._client.chat.completions.create(
            model=self.model, messages=messages  # type: ignore[arg-type]
        )
        return resp.choices[0].message.content or ""


class AnthropicAdapter:
    """
    Adapter for the official ``anthropic`` Python SDK.

    Install: ``pip install anthropic``
    """

    def __init__(
        self,
        api_key: str,
        model: str = "claude-3-5-sonnet-20241022",
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> None:
        try:
            import anthropic as _anthropic
        except ImportError as e:
            raise ImportError("anthropic package not installed. Run: pip install anthropic") from e
        self._client = _anthropic.Anthropic(api_key=api_key, **kwargs)
        self.model = model
        self.max_tokens = max_tokens

    def chat(self, messages: List[Dict[str, str]]) -> str:
        # Anthropic separates system from user/assistant turns
        system = ""
        filtered: List[Dict[str, str]] = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                filtered.append(m)
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=filtered,  # type: ignore[arg-type]
        )
        return resp.content[0].text if resp.content else ""


class GeminiAdapter:
    """
    Adapter for Google Gemini via the ``google-genai`` SDK (v1+).

    Install: ``pip install google-genai``

    Default model is ``gemini-2.5-flash`` (free tier, high rate limits).
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.5-flash",
        **kwargs: Any,
    ) -> None:
        try:
            from google import genai as _genai
            from google.genai import types as _types
        except ImportError as e:
            raise ImportError(
                "google-genai package not installed. Run: pip install google-genai"
            ) from e
        self._client = _genai.Client(api_key=api_key)
        self._types = _types
        self.model = model

    def chat(self, messages: List[Dict[str, str]]) -> str:
        """
        Convert OpenAI-style messages to Gemini Content objects.

        * ``system`` messages become ``system_instruction``
        * ``user`` / ``assistant`` messages become the conversation turns
        """
        from google.genai import types as _types

        system_parts: List[str] = []
        contents: List[Any] = []

        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                system_parts.append(content)
            elif role == "assistant":
                contents.append(
                    _types.Content(role="model", parts=[_types.Part(text=content)])
                )
            else:
                contents.append(
                    _types.Content(role="user", parts=[_types.Part(text=content)])
                )

        # Ensure conversation ends with a user turn
        if not contents or contents[-1].role != "user":
            contents.append(
                _types.Content(role="user", parts=[_types.Part(text="Continue.")])
            )

        config = _types.GenerateContentConfig(
            system_instruction="\n".join(system_parts) if system_parts else None,
            temperature=0.7,
            max_output_tokens=1024,
        )

        resp = self._client.models.generate_content(
            model=self.model,
            contents=contents,
            config=config,
        )
        return resp.text or ""


class OllamaAdapter:
    """
    Adapter for Ollama local models via the ``ollama`` Python SDK.

    Install: ``pip install ollama``
    Requires Ollama server running at ``http://localhost:11434``.
    """

    def __init__(
        self,
        model: str = "llama3",
        host: str = "http://localhost:11434",
        **kwargs: Any,
    ) -> None:
        try:
            import ollama as _ollama
        except ImportError as e:
            raise ImportError("ollama package not installed. Run: pip install ollama") from e
        self._ollama = _ollama
        self.model = model
        self.host = host
        self._kwargs = kwargs

    def chat(self, messages: List[Dict[str, str]]) -> str:
        resp = self._ollama.chat(model=self.model, messages=messages, **self._kwargs)
        return resp["message"]["content"]


class CohereAdapter:
    """
    Adapter for the Cohere ``cohere`` Python SDK.

    Install: ``pip install cohere``
    """

    def __init__(
        self,
        api_key: str,
        model: str = "command-r-plus",
        **kwargs: Any,
    ) -> None:
        try:
            import cohere as _cohere
        except ImportError as e:
            raise ImportError("cohere package not installed. Run: pip install cohere") from e
        self._client = _cohere.Client(api_key=api_key, **kwargs)
        self.model = model

    def chat(self, messages: List[Dict[str, str]]) -> str:
        # Cohere uses preamble + chat_history
        preamble = ""
        history = []
        last_user = ""
        for m in messages:
            if m["role"] == "system":
                preamble = m["content"]
            elif m["role"] == "user":
                last_user = m["content"]
            elif m["role"] == "assistant":
                history.append({"role": "CHATBOT", "message": m["content"]})
        resp = self._client.chat(
            model=self.model,
            message=last_user,
            preamble=preamble,
            chat_history=history,
        )
        return resp.text


class MistralAdapter:
    """
    Adapter for the Mistral AI ``mistralai`` Python SDK.

    Install: ``pip install mistralai``
    """

    def __init__(
        self,
        api_key: str,
        model: str = "mistral-large-latest",
        **kwargs: Any,
    ) -> None:
        try:
            from mistralai import Mistral as _Mistral
        except ImportError as e:
            raise ImportError("mistralai package not installed. Run: pip install mistralai") from e
        self._client = _Mistral(api_key=api_key, **kwargs)
        self.model = model

    def chat(self, messages: List[Dict[str, str]]) -> str:
        resp = self._client.chat.complete(
            model=self.model,
            messages=messages,  # type: ignore[arg-type]
        )
        return resp.choices[0].message.content or ""


class AzureOpenAIAdapter:
    """
    Adapter for Azure OpenAI via the ``openai`` SDK.

    Install: ``pip install openai``
    """

    def __init__(
        self,
        api_key: str,
        azure_endpoint: str,
        api_version: str,
        deployment_name: str,
        **kwargs: Any,
    ) -> None:
        try:
            import openai as _openai
        except ImportError as e:
            raise ImportError("openai package not installed. Run: pip install openai") from e
        self._client = _openai.AzureOpenAI(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
            **kwargs,
        )
        self.deployment_name = deployment_name

    def chat(self, messages: List[Dict[str, str]]) -> str:
        resp = self._client.chat.completions.create(
            model=self.deployment_name,
            messages=messages,  # type: ignore[arg-type]
        )
        return resp.choices[0].message.content or ""
