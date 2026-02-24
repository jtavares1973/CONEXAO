# ============================================================
# gemini_provider.py
# Provedor Google Gemini — via Google AI Studio (GRÁTIS!)
#
# Por que Gemini?
#   - Gemini 2.5 Pro é 100% GRATUITO no free tier do AI Studio
#   - Janela de contexto gigante: 1.048.576 tokens
#   - Qualidade state-of-the-art (topo do LMArena!)
#   - Compatível com o SDK openai — só muda base_url e api_key
#   - Suporta rotação de múltiplas keys (cada conta = mais cota)
#
# Como conseguir sua key gratuita:
#   1. Acesse https://aistudio.google.com/apikey
#   2. Faça login com conta Google (sem cartão de crédito)
#   3. Clique em "Create API Key"
#   4. Copie e coloque no .env como GEMINI_API_KEY
#
# Limites gratuitos (fev/2026):
#   gemini-2.5-pro            → 5 RPM   |  25 RPD  (1M ctx, GRÁTIS!)
#   gemini-2.5-flash          → 15 RPM  | 500 RPD  (1M ctx, GRÁTIS!)
#   gemini-2.5-flash-lite     → 30 RPM  | 1500 RPD (1M ctx, GRÁTIS!)
#   gemini-2.0-flash          → 15 RPM  | 1500 RPD (1M ctx, GRÁTIS!)
#
# Nota: dados são usados para melhorar produtos Google no free tier.
# No plano pago isso não ocorre.
# ============================================================

import logging
import os
import time
from dataclasses import dataclass
from typing import Optional
from openai import OpenAI, RateLimitError, APIStatusError, APIConnectionError

logger = logging.getLogger(__name__)

# URL base do Google AI Studio (compatível com SDK openai)
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

# Modelo padrão: Gemini 2.5 Pro — grátis e poderoso
GEMINI_MODELO_PADRAO = "gemini-2.5-pro"

# Modelos gratuitos disponíveis
GEMINI_MODELOS = {
    # nome                       RPM   RPD     contexto
    "gemini-2.5-pro":          {"rpm": 5,  "rpd":  25,   "ctx": 1048576},
    "gemini-2.5-flash":        {"rpm": 15, "rpd": 500,   "ctx": 1048576},
    "gemini-2.5-flash-lite":   {"rpm": 30, "rpd": 1500,  "ctx": 1048576},
    "gemini-2.0-flash":        {"rpm": 15, "rpd": 1500,  "ctx": 1048576},
    "gemini-2.0-flash-lite":   {"rpm": 30, "rpd": 1500,  "ctx": 1048576},
}

# Cooldown quando bate rate limit por minuto
COOLDOWN_RPM = 65   # segundos (margem extra)


@dataclass
class _StatusKey:
    """Controla o estado de uma API key no pool de rotação."""
    key: str
    modelo: str
    cooldown_ate: float = 0.0
    reqs_hoje: int = 0

    def em_cooldown(self) -> bool:
        return time.time() < self.cooldown_ate

    def aplicar_cooldown(self, segundos: int = COOLDOWN_RPM) -> None:
        self.cooldown_ate = time.time() + segundos
        logger.warning(
            "Gemini key ...%s em cooldown por %ds.", self.key[-6:], segundos
        )

    def resetar(self) -> None:
        self.cooldown_ate = 0.0


class PoolGemini:
    """
    Pool de API keys do Google Gemini com Round-Robin.

    Com uma única key gratuita você já tem acesso ao
    Gemini 2.5 Pro — o modelo mais inteligente gratuitamente disponível.

    Com 2-3 keys (mais contas Google gratuitas) a capacidade multiplica:
      key1 → gemini-2.5-pro   (25 req/dia / 5 RPM)
      key2 → gemini-2.5-flash (500 req/dia / 15 RPM) ← mais volume
      key3 → gemini-2.0-flash (1500 req/dia / 15 RPM) ← alto volume
    """

    def __init__(self, keys_e_modelos: list[tuple[str, str]]):
        """
        Recebe lista de (api_key, modelo).
        Ex: [("AIzaSy...", "gemini-2.5-pro"),
             ("AIzaSy...", "gemini-2.5-flash")]
        """
        self._pool = [
            _StatusKey(key=k, modelo=m)
            for k, m in keys_e_modelos
        ]
        self._indice = 0
        self._clientes: dict[str, OpenAI] = {}

        # Pré-cria um cliente por key
        for sk in self._pool:
            self._clientes[sk.key] = OpenAI(
                api_key=sk.key,
                base_url=GEMINI_BASE_URL,
            )

        logger.info(
            "PoolGemini iniciado com %d key(s). Modelos: %s",
            len(self._pool),
            [s.modelo for s in self._pool],
        )

    @classmethod
    def do_env(cls, modelo: str = GEMINI_MODELO_PADRAO) -> "PoolGemini":
        """
        Cria pool lendo as keys do .env automaticamente.
        Procura GEMINI_API_KEY, GEMINI_API_KEY_2, GEMINI_API_KEY_3, ...
        """
        from .config import carregar_env
        carregar_env()
        keys = []

        # Key principal
        k1 = os.getenv("GEMINI_API_KEY", "")
        if k1:
            keys.append((k1, modelo))

        # Keys extras (rotação — cada conta Google = mais cota)
        for i in range(2, 10):
            k = os.getenv(f"GEMINI_API_KEY_{i}", "")
            if k:
                keys.append((k, modelo))

        if not keys:
            raise ValueError(
                "Nenhuma GEMINI_API_KEY encontrada no .env!\n"
                "Crie uma key gratuita em https://aistudio.google.com/apikey\n"
                "e adicione GEMINI_API_KEY=AIza... no arquivo .env"
            )

        logger.info("PoolGemini: %d key(s) carregadas do .env.", len(keys))
        return cls(keys)

    def completar(
        self,
        mensagens: list[dict],
        max_tokens: int = 2048,
        temperatura: float = 0.7,
    ) -> tuple[str, str]:
        """
        Faz uma chamada com Round-Robin entre as keys.
        Retorna (resposta, modelo_usado).
        Tenta todas as keys antes de desistir.
        """
        total = len(self._pool)
        ultimo_erro: Optional[Exception] = None

        for _ in range(total * 2):  # *2 para dar chance após cooldown
            sk = self._proxima_disponivel()
            if sk is None:
                raise RuntimeError("Todas as keys do Gemini em cooldown.")

            cliente = self._clientes[sk.key]
            try:
                resp = cliente.chat.completions.create(
                    model=sk.modelo,
                    messages=mensagens,     # type: ignore[arg-type]
                    max_tokens=max_tokens,
                    temperature=temperatura,
                )
                sk.reqs_hoje += 1
                conteudo = resp.choices[0].message.content or ""
                logger.debug(
                    "Gemini OK | key=...%s | modelo=%s | tokens=%s",
                    sk.key[-6:], sk.modelo, resp.usage,
                )
                return conteudo.strip(), sk.modelo

            except RateLimitError as exc:
                logger.warning(
                    "Gemini rate limit | key=...%s: %s",
                    sk.key[-6:], exc,
                )
                sk.aplicar_cooldown()
                ultimo_erro = exc

            except (APIStatusError, APIConnectionError) as exc:
                logger.error(
                    "Gemini erro de API | key=...%s: %s", sk.key[-6:], exc
                )
                ultimo_erro = exc

        raise RuntimeError(
            f"Todas as keys Gemini falharam. Último erro: {ultimo_erro}"
        )

    def status(self) -> list[dict]:
        """Retorna situação atual de cada key."""
        agora = time.time()
        return [
            {
                "key_sufixo":  f"...{s.key[-6:]}",
                "modelo":      s.modelo,
                "disponivel":  not s.em_cooldown(),
                "cooldown_s":  max(0, round(s.cooldown_ate - agora)),
                "reqs_hoje":   s.reqs_hoje,
            }
            for s in self._pool
        ]

    def _proxima_disponivel(self) -> Optional["_StatusKey"]:
        total = len(self._pool)
        for _ in range(total):
            sk = self._pool[self._indice % total]
            self._indice = (self._indice + 1) % total
            if not sk.em_cooldown():
                return sk

        # Todas em cooldown — espera a mais próxima e retorna
        mais_cedo = min(self._pool, key=lambda s: s.cooldown_ate)
        espera = max(0.0, mais_cedo.cooldown_ate - time.time())
        if espera > 0:
            logger.warning(
                "Todas as keys Gemini em cooldown. Aguardando %.0fs...", espera
            )
            time.sleep(espera + 1)
        mais_cedo.resetar()
        return mais_cedo
