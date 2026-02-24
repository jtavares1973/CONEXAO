"""
testar_gemini.py
================
Testa o AgenteGemini com o Gemini 2.5 Pro (GRÁTIS via Google AI Studio).

Execute:
    .venv\Scripts\python.exe testar_gemini.py
"""

import time
from dotenv import load_dotenv

load_dotenv()

from openai_agent import AgenteGemini


def teste_chat():
    """Teste básico de chat — verifica conectividade e velocidade."""
    print("\n" + "="*55)
    print("  TESTE: Gemini 2.5 Pro via Google AI Studio (FREE)")
    print("="*55)

    agente = AgenteGemini(modelo="gemini-2.5-pro")

    perguntas = [
        "Qual é a capital do Brasil? Responda em 1 linha.",
        "Quanto é 17 x 23? Só o número.",
        "Cite 3 linguagens de programação populares em 1 linha.",
    ]

    for i, pergunta in enumerate(perguntas, 1):
        print(f"\n  Req #{i}: {pergunta}")
        inicio = time.time()
        resp = agente.executar("chat", pergunta)
        dur = time.time() - inicio
        print(f"  → {resp[:80]}")
        print(f"  ⏱  {dur:.1f}s")

    print("\n  Status do pool:")
    for s in agente.status_pool():
        status = "✅ livre" if s["disponivel"] else f"⏳ cooldown {s['cooldown_s']}s"
        print(f"    key ...{s['key_sufixo'][-6:]} | {s['modelo']} | {status} | {s['reqs_hoje']} req")


def teste_analise():
    """Testa a Skill de análise — mostra o poder do Gemini 2.5 Pro."""
    print("\n" + "="*55)
    print("  TESTE: Análise de texto (foco=riscos)")
    print("="*55)

    agente = AgenteGemini(modelo="gemini-2.5-pro")

    texto = """
    A empresa XYZ registrou crescimento de 45% no faturamento em 2025,
    impulsionado pela expansão para o mercado latinoamericano. No entanto,
    a dívida líquida aumentou 3x e a margem EBITDA caiu de 22% para 14%.
    O CEO anunciou planos de IPO para o 2T 2026, apesar da instabilidade
    nos mercados emergentes e da taxa Selic em 14,75%.
    """

    inicio = time.time()
    resp = agente.executar("analise", texto, foco="riscos")
    dur = time.time() - inicio

    print(f"\n  Resposta ({dur:.1f}s):\n")
    print(resp[:500])
    if len(resp) > 500:
        print(f"  ... [+{len(resp)-500} chars]")


def teste_flash_alto_volume():
    """Demonstra o Gemini 2.5 Flash para processamento de maior volume."""
    print("\n" + "="*55)
    print("  TESTE: Gemini 2.5 Flash (15 RPM / 500 RPD)")
    print("="*55)

    agente = AgenteGemini(modelo="gemini-2.5-flash")

    inicio = time.time()
    resp = agente.executar("resumo", """
        Inteligência artificial tem revolucionado diversas indústrias ao redor
        do mundo. Desde diagnósticos médicos mais precisos até veículos autônomos,
        a IA está presente em praticamente todos os setores. No campo educacional,
        sistemas de tutoria personalizada adaptam o conteúdo ao ritmo de cada
        aluno. Na indústria, robôs inteligentes trabalham ao lado de humanos
        aumentando a produtividade. O desafio permanece em garantir que os
        benefícios sejam distribuídos equitativamente na sociedade.
    """, tamanho="curto")
    dur = time.time() - inicio

    print(f"\n  Resumo ({dur:.1f}s):\n  {resp}")


if __name__ == "__main__":
    print("\n🧪 Testando integração com Google Gemini AI Studio...")
    print("   Modelo padrão: gemini-2.5-pro (GRÁTIS!)")

    try:
        teste_chat()
        time.sleep(2)   # breve pausa entre blocos
        teste_analise()
        time.sleep(2)
        teste_flash_alto_volume()
        print("\n✅ Todos os testes concluídos com sucesso!")
        print("\n📊 Resumo de capacidade disponível (free tier):")
        print("   gemini-2.5-pro        →  5 RPM |   25 req/dia")
        print("   gemini-2.5-flash      → 15 RPM |  500 req/dia")
        print("   gemini-2.0-flash      → 15 RPM | 1500 req/dia")
        print("   gemini-2.5-flash-lite → 30 RPM | 1500 req/dia")
        print("\n💡 Adicionar GEMINI_API_KEY_2 no .env dobra toda a capacidade!")
    except Exception as e:
        print(f"\n❌ Erro: {e}")
        raise
