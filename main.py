# ============================================================
# main.py  —  Script principal de demonstração
#
# Como usar:
#   1. Instale as dependências:  pip install -r requirements.txt
#   2. Configure o .env com sua OPENROUTER_API_KEY
#   3. Execute:  python main.py
# ============================================================

import logging
import os
from dotenv import load_dotenv

# Carrega as variáveis do arquivo .env automaticamente
load_dotenv()

# ── Configuração de logging ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Importa o pacote do agente ───────────────────────────────────────────────
from openai_agent import AgenteOpenAI


# ============================================================
# TEXTOS DE EXEMPLO PARA AS DEMOS
# ============================================================

TEXTO_PARA_RESUMO = """
A inteligência artificial está transformando profundamente a maneira como as empresas 
operam e como as pessoas interagem com a tecnologia. Modelos de linguagem como GPT, 
Claude e Gemini demonstraram capacidade excepcional em tarefas de compreensão, geração 
de texto, codificação e raciocínio complexo. Essas ferramentas estão sendo adotadas em 
diversas áreas: saúde, educação, finanças, engenharia e entretenimento. No entanto, 
desafios éticos permanecem, incluindo viés algorítmico, privacidade de dados, 
desinformação e impacto no mercado de trabalho. Pesquisadores e reguladores buscam 
equilibrar a inovação com a proteção dos usuários, criando frameworks de IA responsável 
que garantam segurança, transparência e equidade nos sistemas automatizados.
"""

TEXTO_PARA_ANALISE = """
Vendas Q1: R$ 450.000 (+12% vs Q1 anterior)
Vendas Q2: R$ 510.000 (+13% vs Q2 anterior)
Vendas Q3: R$ 480.000 (-6% vs Q2)
Clientes novos: 340 (meta era 400)
Churn rate: 8% (meta era 5%)
NPS: 72 (meta era 75)
Produto mais vendido: Plano Premium (+40% de adesões)
Região com melhor performance: Sudeste (58% do total)
"""

TEXTO_PARA_TRADUCAO = """
Artificial intelligence is revolutionizing software development by automating 
repetitive tasks, improving code quality, and enabling developers to focus on 
creative problem-solving. Tools like GitHub Copilot and ChatGPT have become 
essential companions for modern developers.
"""


# ============================================================
# FUNÇÕES DE DEMONSTRAÇÃO
# ============================================================

def demo_chat(agente: AgenteOpenAI) -> None:
    """Demonstra conversa com histórico usando a SkillChat."""
    print("\n" + "="*60)
    print("💬  DEMO: Chat com Histórico")
    print("="*60)

    perguntas = [
        "Quais são os 3 principais benefícios de usar modelos de linguagem em empresas?",
        "E quais são os principais riscos associados?",
        "Com base no que você disse, qual seria sua recomendação geral?",
    ]

    for pergunta in perguntas:
        print(f"\n🙋 Usuário: {pergunta}")
        resposta = agente.executar("chat", pergunta)
        print(f"🤖 Agente:  {resposta}")


def demo_resumo(agente: AgenteOpenAI) -> None:
    """Demonstra resumo com diferentes tamanhos."""
    print("\n" + "="*60)
    print("📄  DEMO: Resumo de Texto")
    print("="*60)

    for tamanho in ["curto", "médio"]:
        print(f"\n📏 Tamanho do resumo: {tamanho}")
        resumo = agente.executar("resumo", TEXTO_PARA_RESUMO, tamanho=tamanho)
        print(f"📝 {resumo}")


def demo_analise(agente: AgenteOpenAI) -> None:
    """Demonstra análise de dados de vendas."""
    print("\n" + "="*60)
    print("📊  DEMO: Análise de Dados")
    print("="*60)

    for foco in ["tendências", "riscos"]:
        print(f"\n🔍 Foco da análise: {foco}")
        analise = agente.executar("analise", TEXTO_PARA_ANALISE, foco=foco)
        print(f"📌 {analise}")


def demo_traducao(agente: AgenteOpenAI) -> None:
    """Demonstra tradução automática."""
    print("\n" + "="*60)
    print("🌐  DEMO: Tradução Automática")
    print("="*60)

    print(f"\n🔤 Texto original (inglês):\n{TEXTO_PARA_TRADUCAO.strip()}")
    traduzido = agente.executar(
        "traducao",
        TEXTO_PARA_TRADUCAO,
        idioma_alvo="português do Brasil",
    )
    print(f"\n🇧🇷 Tradução:\n{traduzido}")


def exibir_info_agente(agente: AgenteOpenAI) -> None:
    """Exibe informações sobre o agente e modelo selecionado."""
    print("\n" + "="*60)
    print("🤖  INFORMAÇÕES DO AGENTE")
    print("="*60)
    print(f"  Modelo selecionado : {agente.model_id}")
    print(f"  Skills disponíveis : {', '.join(agente.skills_disponiveis())}")
    print(f"  API base URL       : {agente._base_url}")
    print("="*60)


# ============================================================
# PONTO DE ENTRADA
# ============================================================

def main() -> None:
    logger.info("Iniciando agente...")

    # Instancia o agente — seleciona automaticamente o melhor modelo gratuito
    agente = AgenteOpenAI(apenas_gratuitos=True)

    # Exibe informações do agente
    exibir_info_agente(agente)

    # Executa as demos
    demo_chat(agente)
    demo_resumo(agente)
    demo_analise(agente)
    demo_traducao(agente)

    print("\n✅ Todas as demos concluídas com sucesso!")


if __name__ == "__main__":
    main()
