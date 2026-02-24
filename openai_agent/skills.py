# ============================================================
# skills.py
# Cada "Skill" (habilidade) encapsula um tipo específico de
# tarefa que o agente pode executar usando a API.
#
# Estrutura:
#   SkillBase       — classe abstrata / contrato comum
#   SkillChat       — conversa livre com histórico
#   SkillResumo     — resumo de texto longo
#   SkillAnalise    — análise estruturada de dados/texto
#   SkillTradução   — tradução entre idiomas
# ============================================================

import logging
from abc import ABC, abstractmethod
from typing import Optional
from openai import OpenAI

logger = logging.getLogger(__name__)

# ── Tipo de mensagem do histórico ────────────────────────────────────────────
Mensagem = dict[str, str]  # {"role": "user"|"assistant"|"system", "content": "..."}


# ============================================================
# CLASSE BASE — Contrato obrigatório para todas as skills
# ============================================================
class SkillBase(ABC):
    """
    Classe abstrata que define o contrato de uma Skill.
    Toda skill recebe um client OpenAI e um model_id já selecionado.
    """

    nome: str = "skill_base"
    descricao: str = "Habilidade genérica"

    def __init__(self, client: OpenAI, model_id: str, temperatura: float = 0.7):
        self.client = client
        self.model_id = model_id
        self.temperatura = temperatura

    @abstractmethod
    def executar(self, entrada: str, **kwargs) -> str:
        """Executa a skill e retorna o resultado como string."""
        ...

    # Método utilitário compartilhado entre todas as skills
    def _chamar_api(
        self,
        mensagens: list[Mensagem],
        max_tokens: int = 2048,
        temperatura: Optional[float] = None,
    ) -> str:
        """
        Faz a chamada real à API e retorna o conteúdo da resposta.
        Se o modelo não suportar system prompt (ex: Gemma via Google AI Studio),
        funde automaticamente a mensagem de sistema na mensagem do usuário e retenta.
        Lança exceção em caso de falha para o agente tratar o fallback.
        """
        from openai import BadRequestError

        def _fazer_chamada(msgs: list[Mensagem]) -> str:
            resp = self.client.chat.completions.create(
                model=self.model_id,
                messages=msgs,            # type: ignore[arg-type]
                max_tokens=max_tokens,
                temperature=temperatura if temperatura is not None else self.temperatura,
            )
            conteudo = resp.choices[0].message.content or ""
            logger.debug("[%s] tokens usados: %s", self.nome, resp.usage)
            return conteudo.strip()

        try:
            return _fazer_chamada(mensagens)
        except BadRequestError as exc:
            # Alguns modelos (ex: Gemma via Google AI Studio) não aceitam system prompt.
            # Fusiona sistema+usuário em uma única mensagem e retenta.
            msg_str = str(exc).lower()
            if "developer instruction" in msg_str or "system" in msg_str:
                logger.warning(
                    "[%s] Modelo '%s' não suporta system prompt — fundindo em user.",
                    self.nome, self.model_id,
                )
                mensagens_sem_system = self._fundir_system_em_user(mensagens)
                return _fazer_chamada(mensagens_sem_system)
            raise

    @staticmethod
    def _fundir_system_em_user(mensagens: list[Mensagem]) -> list[Mensagem]:
        """
        Converte mensagens que possuem 'role=system' para 'role=user',
        prefixando o conteúdo ao próximo bloco de usuário.
        Compatibilidade com modelos que não aceitam system prompt.
        """
        resultado: list[Mensagem] = []
        pendente_system = ""
        for msg in mensagens:
            if msg["role"] == "system":
                pendente_system += msg["content"] + "\n\n"
            elif msg["role"] == "user" and pendente_system:
                resultado.append(
                    {"role": "user", "content": pendente_system + msg["content"]}
                )
                pendente_system = ""
            else:
                resultado.append(msg)
        return resultado


# ============================================================
# SKILL: Chat com histórico de conversa
# ============================================================
class SkillChat(SkillBase):
    """
    Conversa livre com o modelo.
    Mantém histórico interno para conversas de múltiplos turnos.
    """

    nome = "chat"
    descricao = "Conversa livre com histórico de mensagens"

    def __init__(self, client: OpenAI, model_id: str, system_prompt: str = ""):
        super().__init__(client, model_id)
        self._historico: list[Mensagem] = []

        # Instrução de sistema (persona/contexto)
        self._system = system_prompt or (
            "Você é um assistente inteligente e prestativo. "
            "Responda sempre em português do Brasil de forma clara e objetiva."
        )

    def executar(self, entrada: str, **kwargs) -> str:
        """Envia mensagem do usuário e retorna resposta do modelo."""
        self._historico.append({"role": "user", "content": entrada})

        mensagens: list[Mensagem] = (
            [{"role": "system", "content": self._system}]
            + self._historico
        )

        resposta = self._chamar_api(mensagens, max_tokens=kwargs.get("max_tokens", 2048))
        self._historico.append({"role": "assistant", "content": resposta})
        return resposta

    def limpar_historico(self) -> None:
        """Reinicia o histórico de conversa."""
        self._historico.clear()
        logger.info("[SkillChat] Histórico limpo.")


# ============================================================
# SKILL: Resumo de texto
# ============================================================
class SkillResumo(SkillBase):
    """
    Gera um resumo conciso de um texto fornecido.
    Pode receber o tamanho desejado do resumo (curto/médio/longo).
    """

    nome = "resumo"
    descricao = "Resume textos longos de forma clara e objetiva"

    def executar(self, entrada: str, tamanho: str = "médio", **kwargs) -> str:
        """
        Parâmetros:
            entrada  — texto a ser resumido
            tamanho  — 'curto' (3 frases), 'médio' (1 parágrafo), 'longo' (3 parágrafos)
        """
        instrucao_tamanho = {
            "curto": "em no máximo 3 frases",
            "médio": "em um parágrafo coeso",
            "longo": "em até 3 parágrafos detalhados",
        }.get(tamanho, "em um parágrafo coeso")

        mensagens: list[Mensagem] = [
            {
                "role": "system",
                "content": (
                    "Você é um especialista em síntese de informação. "
                    "Resuma textos de forma precisa, sem perder os pontos principais. "
                    "Responda sempre em português do Brasil."
                ),
            },
            {
                "role": "user",
                "content": f"Resuma o texto abaixo {instrucao_tamanho}:\n\n{entrada}",
            },
        ]
        return self._chamar_api(mensagens, max_tokens=kwargs.get("max_tokens", 1024))


# ============================================================
# SKILL: Análise de texto/dados
# ============================================================
class SkillAnalise(SkillBase):
    """
    Realiza análise estruturada de um texto ou conjunto de dados.
    Retorna insights, padrões e conclusões em formato legível.
    """

    nome = "analise"
    descricao = "Analisa textos ou dados e extrai insights estruturados"

    def executar(self, entrada: str, foco: str = "geral", **kwargs) -> str:
        """
        Parâmetros:
            entrada — texto ou dados a analisar
            foco    — 'geral', 'sentimento', 'tendências', 'riscos', 'oportunidades'
        """
        instrucoes_foco = {
            "geral":        "Faça uma análise completa identificando os principais tópicos, padrões e conclusões.",
            "sentimento":   "Analise o sentimento predominante (positivo/negativo/neutro) com justificativa.",
            "tendências":   "Identifique tendências e padrões ao longo do tempo ou do texto.",
            "riscos":       "Liste os principais riscos e pontos de atenção identificados.",
            "oportunidades":"Liste as principais oportunidades e pontos positivos identificados.",
        }.get(foco, "Faça uma análise completa.")

        mensagens: list[Mensagem] = [
            {
                "role": "system",
                "content": (
                    "Você é um analista especialista. "
                    "Apresente sua análise em tópicos claros com títulos. "
                    "Responda sempre em português do Brasil."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"{instrucoes_foco}\n\n"
                    f"Dados/Texto para análise:\n\n{entrada}"
                ),
            },
        ]
        return self._chamar_api(mensagens, max_tokens=kwargs.get("max_tokens", 2048))


# ============================================================
# SKILL: Tradução
# ============================================================
class SkillTradução(SkillBase):
    """
    Traduz um texto de qualquer idioma para o idioma alvo desejado.
    Detecta automaticamente o idioma de origem.
    """

    nome = "traducao"
    descricao = "Traduz textos entre idiomas automaticamente"

    def executar(self, entrada: str, idioma_alvo: str = "português do Brasil", **kwargs) -> str:
        """
        Parâmetros:
            entrada       — texto a traduzir
            idioma_alvo   — idioma de destino (padrão: 'português do Brasil')
        """
        mensagens: list[Mensagem] = [
            {
                "role": "system",
                "content": (
                    "Você é um tradutor profissional especializado. "
                    "Detecte automaticamente o idioma de origem e traduza para o idioma solicitado. "
                    "Preserve o tom, estilo e formatação originais. "
                    "Retorne APENAS o texto traduzido, sem explicações adicionais."
                ),
            },
            {
                "role": "user",
                "content": f"Traduza o texto abaixo para {idioma_alvo}:\n\n{entrada}",
            },
        ]
        return self._chamar_api(
            mensagens,
            max_tokens=kwargs.get("max_tokens", int(len(entrada.split()) * 1.5) + 500),
            temperatura=0.3,  # menor temperatura = tradução mais fiel
        )
