"""
python -m openai_agent <comando>

Comandos disponíveis:
  setup    → Assistente interativo para configurar as chaves de API
  status   → Mostra quais chaves estão configuradas
  test     → Testa todos os provedores configurados
"""

import sys
import os
from pathlib import Path


def cmd_status():
    """Mostra status de todas as chaves."""
    from .config import carregar_env, status_keys, ENV_GLOBAL
    env_path = carregar_env()

    print("\n🔑 Status das chaves de API — naccarona-agent")
    print("=" * 48)
    if env_path:
        print(f"  .env carregado de: {env_path}")
    else:
        print(f"  .env global em: {ENV_GLOBAL}")
    print()
    for chave, st in status_keys().items():
        print(f"  {chave:<28} {st}")
    print()


def cmd_setup():
    """Assistente interativo de configuração."""
    from .config import caminho_env_global

    env_path = caminho_env_global()

    print("\n⚙️  Configuração das chaves — naccarona-agent")
    print("=" * 48)
    print(f"  As chaves serão salvas em:\n  {env_path}")
    print("\n  Deixe em branco para pular um provedor.\n")

    provedores = [
        ("OPENROUTER_API_KEY",  "OpenRouter  (sk-or-v1-...)", "https://openrouter.ai/settings/keys"),
        ("GROQ_API_KEY",        "Groq        (gsk_...)",      "https://console.groq.com"),
        ("GEMINI_API_KEY",      "Gemini      (AIza...)",      "https://aistudio.google.com/apikey"),
        ("CEREBRAS_API_KEY",    "Cerebras    (csk_...)",      "https://cloud.cerebras.ai"),
        ("SAMBANOVA_API_KEY",   "SambaNova   (UUID)",         "https://cloud.sambanova.ai/apis"),
    ]

    novas_linhas = []
    alterou = False

    # Lê valores existentes
    existentes: dict[str, str] = {}
    if env_path.exists():
        for linha in env_path.read_text(encoding="utf-8").splitlines():
            if "=" in linha and not linha.startswith("#"):
                k, _, v = linha.partition("=")
                existentes[k.strip()] = v.strip()

    for var, label, url in provedores:
        atual = existentes.get(var, "")
        sufixo = f"  (atual: ...{atual[-8:]})" if atual else ""
        entrada = input(f"  {label}{sufixo}\n    → ").strip()
        if entrada:
            existentes[var] = entrada
            alterou = True
        novas_linhas.append(f'{var}={existentes.get(var, "")}')
        # Keys extras (2, 3...)
        for i in range(2, 4):
            var_extra = f"{var}_{i}"
            atual_extra = existentes.get(var_extra, "")
            sufixo_extra = f"  (atual: ...{atual_extra[-8:]})" if atual_extra else ""
            entrada_extra = input(
                f"  {label} #{i} (opcional){sufixo_extra}\n    → "
            ).strip()
            if entrada_extra:
                existentes[var_extra] = entrada_extra
                novas_linhas.append(f"{var_extra}={entrada_extra}")
                alterou = True
            elif atual_extra:
                novas_linhas.append(f"{var_extra}={atual_extra}")
        print()

    if alterou:
        conteudo = "\n".join(novas_linhas) + "\n"
        env_path.write_text(conteudo, encoding="utf-8")
        print(f"✅ Chaves salvas em {env_path}")
    else:
        print("ℹ️  Nenhuma alteração feita.")

    print("\nAgora você pode usar em qualquer projeto:")
    print("  from openai_agent import AgenteGemini, AgenteCerebras")
    print("  agente = AgenteGemini()")
    print("  resp = agente.executar('chat', 'Olá!')\n")


def cmd_test():
    """Testa rapidamente todos os provedores configurados."""
    from .config import carregar_env, status_keys
    carregar_env()

    print("\n🧪 Testando provedores configurados...\n")

    pergunta = "Responda apenas: OK"

    testes = [
        ("Gemini",    "GEMINI_API_KEY",    "openai_agent.gemini_agent",    "AgenteGemini",    {"modelo": "gemini-2.5-flash"}),
        ("Groq",      "GROQ_API_KEY",      "openai_agent.groq_agent",      "AgenteGroq",      {}),
        ("Cerebras",  "CEREBRAS_API_KEY",  "openai_agent.cerebras_agent",  "AgenteCerebras",  {}),
        ("SambaNova", "SAMBANOVA_API_KEY", "openai_agent.sambanova_agent", "AgenteSambaNova", {}),
        ("OpenRouter","OPENROUTER_API_KEY","openai_agent.agent",           "AgenteOpenAI",    {}),
    ]

    import time, importlib
    for nome, var, modulo, classe, kwargs in testes:
        if not os.getenv(var, "").strip():
            print(f"  {nome:<12} ⏭  sem key")
            continue
        try:
            mod = importlib.import_module(modulo)
            agente = getattr(mod, classe)(**kwargs)
            inicio = time.time()
            resp = agente.executar("chat", pergunta)
            dur = time.time() - inicio
            print(f"  {nome:<12} ✅  {resp[:30]:<30}  ({dur:.1f}s)")
        except Exception as e:
            print(f"  {nome:<12} ❌  {str(e)[:60]}")

    print()


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"

    if cmd == "setup":
        cmd_setup()
    elif cmd == "test":
        cmd_test()
    elif cmd == "status":
        cmd_status()
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
