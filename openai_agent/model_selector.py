# ============================================================
# model_selector.py
# Responsável por:
#   1. Consultar modelos disponíveis via API
#   2. Filtrar modelos gratuitos ou de baixo custo
#   3. Ranquear e selecionar o melhor modelo automaticamente
# ============================================================

import logging
from dataclasses import dataclass, field
from typing import Optional
from openai import OpenAI

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Modelos conhecidos no OpenRouter com metadados de custo/contexto/data
# (usado como fallback ou para enriquecer a seleção)
# Os valores de "custo" são em USD por 1M tokens (prompt/completion).
# Gratuitos têm custo 0.
# ---------------------------------------------------------------------------
MODELOS_CONHECIDOS: dict[str, dict] = {
    # ── Gratuitos ────────────────────────────────────────────────────────────
    # Nota: Gemma via Google AI Studio não suporta system prompt — removido da lista
    # Modelos removidos por indisponibilidade (404): mistral-7b:free, llama-3.1-8b:free
    "mistralai/mistral-small-3.1-24b-instruct:free": {"custo": 0.0, "contexto": 128000, "ano": 2025},
    "meta-llama/llama-3.3-70b-instruct:free":   {"custo": 0.0, "contexto": 131072, "ano": 2025},
    "deepseek/deepseek-chat-v3-0324:free":      {"custo": 0.0, "contexto": 163840, "ano": 2025},
    "deepseek/deepseek-r1:free":                {"custo": 0.0, "contexto": 163840, "ano": 2025},
    "qwen/qwen3-235b-a22b:free":                {"custo": 0.0, "contexto": 131072, "ano": 2025},
    "qwen/qwen3-30b-a3b:free":                  {"custo": 0.0, "contexto": 131072, "ano": 2025},
    "qwen/qwen3-14b:free":                      {"custo": 0.0, "contexto": 40960,  "ano": 2025},
    # ── Baixo custo (< USD 0,20 / 1M tokens) ────────────────────────────────
    "openai/gpt-4o-mini":                       {"custo": 0.15, "contexto": 128000, "ano": 2024},
    "anthropic/claude-haiku":                   {"custo": 0.25, "contexto": 200000, "ano": 2024},
}

# Custo máximo aceito como "baixo custo" (USD / 1M tokens)
LIMITE_CUSTO_BAIXO = 0.30


@dataclass
class ModeloInfo:
    """Representa um modelo candidato com seus atributos de seleção."""
    id: str
    custo: float = 0.0          # USD por 1M tokens (prompt)
    contexto: int = 4096        # tamanho máximo do contexto em tokens
    ano: int = 2020             # ano de lançamento aproximado
    disponivel: bool = True     # confirmado via API
    score: float = field(init=False, default=0.0)

    def calcular_score(self) -> None:
        """
        Pontuação composta (maior = melhor):
          - Bônus por ser gratuito: +100
          - Penalidade por custo:   -custo * 50
          - Bônus por contexto:     log2(contexto) * 5
          - Bônus por ano recente:  (ano - 2020) * 10
        """
        import math
        bonus_gratis   = 100 if self.custo == 0 else 0
        penalidade     = self.custo * 50
        bonus_contexto = math.log2(max(self.contexto, 1)) * 5
        bonus_ano      = (self.ano - 2020) * 10
        self.score = bonus_gratis - penalidade + bonus_contexto + bonus_ano


class SeletorDeModelos:
    """
    Seleciona automaticamente o melhor modelo disponível na conta OpenRouter.

    Fluxo:
      1. Busca lista de modelos via client.models.list()
      2. Filtra apenas modelos gratuitos/baixo custo
      3. Enriquece com metadados conhecidos
      4. Calcula score e retorna o melhor
    """

    def __init__(self, client: OpenAI, apenas_gratuitos: bool = True):
        self.client = client
        self.apenas_gratuitos = apenas_gratuitos
        self._cache_modelos: list[ModeloInfo] = []

    # ------------------------------------------------------------------
    # Método público principal
    # ------------------------------------------------------------------
    def selecionar_melhor(self) -> str:
        """Retorna o ID do melhor modelo disponível."""
        candidatos = self._obter_candidatos()
        if not candidatos:
            logger.warning("Nenhum modelo candidato encontrado — usando fallback.")
            return self._fallback()

        melhor = max(candidatos, key=lambda m: m.score)
        logger.info(
            "Modelo selecionado: %s | score=%.1f | custo=%.3f | contexto=%d",
            melhor.id, melhor.score, melhor.custo, melhor.contexto,
        )
        return melhor.id

    def listar_candidatos(self) -> list[ModeloInfo]:
        """Retorna lista ordenada de candidatos (melhor primeiro)."""
        candidatos = self._obter_candidatos()
        return sorted(candidatos, key=lambda m: m.score, reverse=True)

    # ------------------------------------------------------------------
    # Métodos internos
    # ------------------------------------------------------------------
    def _obter_candidatos(self) -> list[ModeloInfo]:
        if self._cache_modelos:
            return self._cache_modelos

        ids_disponiveis = self._buscar_ids_da_api()
        candidatos: list[ModeloInfo] = []

        for modelo_id in ids_disponiveis:
            meta = MODELOS_CONHECIDOS.get(modelo_id, {})
            custo = meta.get("custo", 0.0)

            # Aplica filtro de custo
            if self.apenas_gratuitos and custo > 0:
                continue
            if not self.apenas_gratuitos and custo > LIMITE_CUSTO_BAIXO:
                continue

            info = ModeloInfo(
                id=modelo_id,
                custo=custo,
                contexto=meta.get("contexto", 4096),
                ano=meta.get("ano", 2020),
            )
            info.calcular_score()
            candidatos.append(info)

        # Se nenhum modelo da API bater com os conhecidos, usa os conhecidos diretamente
        if not candidatos:
            logger.warning(
                "API não retornou modelos conhecidos. "
                "Usando lista interna como candidatos."
            )
            for modelo_id, meta in MODELOS_CONHECIDOS.items():
                custo = meta.get("custo", 0.0)
                if self.apenas_gratuitos and custo > 0:
                    continue
                info = ModeloInfo(
                    id=modelo_id,
                    custo=custo,
                    contexto=meta.get("contexto", 4096),
                    ano=meta.get("ano", 2020),
                )
                info.calcular_score()
                candidatos.append(info)

        self._cache_modelos = candidatos
        return candidatos

    def _buscar_ids_da_api(self) -> list[str]:
        """Consulta a API e retorna lista de IDs de modelos disponíveis."""
        try:
            resposta = self.client.models.list()
            ids = [m.id for m in resposta.data]
            logger.info("API retornou %d modelos.", len(ids))
            return ids
        except Exception as exc:
            logger.error("Falha ao consultar modelos via API: %s", exc)
            return []

    @staticmethod
    def _fallback() -> str:
        """Retorna modelo de fallback seguro."""
        return "mistralai/mistral-7b-instruct:free"
