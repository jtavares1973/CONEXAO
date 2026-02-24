"""
testar_cerebras_sambanova.py
============================
Testa AgenteCerebras e AgenteSambaNova.

Execute:
    .venv\Scripts\python.exe testar_cerebras_sambanova.py

Keys necessárias no .env:
    CEREBRAS_API_KEY=csk_...   → https://cloud.cerebras.ai
    SAMBANOVA_API_KEY=sn_...   → https://cloud.sambanova.ai/apis
"""

import time
from dotenv import load_dotenv

load_dotenv()


def teste_cerebras():
    """Testa o AgenteCerebras — esperado ~2.200 tok/s."""
    from openai_agent import AgenteCerebras

    print("\n" + "="*55)
    print("  TESTE: Cerebras (~2.200 tok/s — o mais rápido!)")
    print("="*55)

    agente = AgenteCerebras(modelo="llama3.1-8b")

    perguntas = [
        "Qual é a capital do Brasil? Responda em 1 linha.",
        "Quanto é 17 x 23? Só o número.",
        "Cite 3 linguagens de programação em 1 linha.",
    ]

    for i, pergunta in enumerate(perguntas, 1):
        print(f"\n  Req #{i}: {pergunta}")
        inicio = time.time()
        resp = agente.executar("chat", pergunta)
        dur = time.time() - inicio
        print(f"  → {resp[:80]}")
        print(f"  ⏱  {dur:.1f}s")

    print("\n  Status pool:")
    for s in agente.status_pool():
        status = "✅ livre" if s["disponivel"] else f"⏳ {s['cooldown_s']}s"
        print(f"    ...{s['key_sufixo'][-6:]} | {s['modelo']} | {status} | {s['reqs_total']} req")


def teste_sambanova():
    """Testa o AgenteSambaNova com DeepSeek V3.2."""
    from openai_agent import AgenteSambaNova

    print("\n" + "="*55)
    print("  TESTE: SambaNova com DeepSeek V3.2")
    print("="*55)

    agente = AgenteSambaNova(modelo="DeepSeek-V3.2")

    print("\n  Req #1: análise de texto")
    inicio = time.time()
    resp = agente.executar("analise",
        "A empresa triplicou o faturamento mas a dívida dobrou e "
        "o CEO foi substituído após escândalos.",
        foco="riscos"
    )
    dur = time.time() - inicio
    print(f"  → {resp[:200]}")
    print(f"  ⏱  {dur:.1f}s")

    print("\n  Status pool:")
    for s in agente.status_pool():
        status = "✅ livre" if s["disponivel"] else f"⏳ {s['cooldown_s']}s"
        print(f"    ...{s['key_sufixo'][-6:]} | {s['modelo']} | {status} | {s['reqs_total']} req")


def teste_sambanova_gpt_oss():
    """Testa o AgenteSambaNova com GPT-OSS 120B (modelo open-source da OpenAI)."""
    from openai_agent import AgenteSambaNova

    print("\n" + "="*55)
    print("  TESTE: SambaNova com GPT-OSS 120B (OpenAI open-source!)")
    print("="*55)

    agente = AgenteSambaNova(modelo="gpt-oss-120b")
    inicio = time.time()
    resp = agente.executar("chat", "Quem criou você? Responda em 1 linha.")
    dur = time.time() - inicio
    print(f"\n  → {resp[:120]}")
    print(f"  ⏱  {dur:.1f}s")


if __name__ == "__main__":
    import os

    tem_cerebras  = bool(os.getenv("CEREBRAS_API_KEY", "").strip())
    tem_sambanova = bool(os.getenv("SAMBANOVA_API_KEY", "").strip())

    print("\n🧪 Testando Cerebras e SambaNova...")

    if not tem_cerebras and not tem_sambanova:
        print("\n⚠️  Nenhuma key encontrada no .env!")
        print("\nPara testar:")
        print("  Cerebras  → https://cloud.cerebras.ai  → CEREBRAS_API_KEY=csk_...")
        print("  SambaNova → https://cloud.sambanova.ai/apis → SAMBANOVA_API_KEY=sn_...")
    else:
        if tem_cerebras:
            try:
                teste_cerebras()
                print("\n✅ Cerebras OK!")
            except Exception as e:
                print(f"\n❌ Cerebras falhou: {e}")
        else:
            print("\n⏭️  Cerebras pulado (sem CEREBRAS_API_KEY no .env)")

        if tem_sambanova:
            try:
                time.sleep(1)
                teste_sambanova()
                time.sleep(1)
                teste_sambanova_gpt_oss()
                print("\n✅ SambaNova OK!")
            except Exception as e:
                print(f"\n❌ SambaNova falhou: {e}")
        else:
            print("\n⏭️  SambaNova pulado (sem SAMBANOVA_API_KEY no .env)")

    print("\n" + "="*55)
    print("  RESUMO — Todos os provedores disponíveis no projeto")
    print("="*55)
    print("  Provedor        | Melhor modelo grátis    | RPM  | Velocidade")
    print("  ----------------+-------------------------+------+-----------")
    print("  OpenRouter      | llama-3.3-70b           |   8  | ~80 tok/s")
    print("  Groq            | llama-3.1-8b-instant    |  30  | ~800 tok/s")
    print("  Google Gemini   | gemini-2.5-pro          |   5  | ~100 tok/s")
    print("  Cerebras        | llama3.1-8b             |  30  | ~2200 tok/s ⚡")
    print("  SambaNova       | DeepSeek-V3.2 / GPT-OSS |  30  | ~500 tok/s")
