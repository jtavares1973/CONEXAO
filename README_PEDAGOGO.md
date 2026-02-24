Objetivo
- Inspecionar `PEDAGOGO.xlsx` e gerar mapeamento dos campos com regras sugeridas de tratamento.

Uso rápido
1. Criar um ambiente virtual (opcional) e instalar dependências:

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

2. Executar a análise:

```bash
python analyze_pedagogo.py
```

Saídas
- `pedagogo_mapping.json`: mapeamento com sugestões por campo
- `pedagogo_sample.csv`: primeiras 200 linhas para inspeção rápida
- `pedagogo_mapping.xlsx`: o mesmo mapeamento em planilha Excel para leitura interativa
- `pedagogo_cleaned.xlsx` / `pedagogo_cleaned.csv`: dados tratados com normalização de nomes, e-mails, CPF/telefone e datas
- As colunas `sede_cidade` e `sede_estado` no dataset limpo trazem a versão padronizada da `Sede` original
