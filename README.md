# Assignment 25 — Prompting and LangChain Chains

**Student:** Patan Khalilulla Khan

**Course:** GenAI Training Program

This assignment demonstrates prompt templates, validated structured output,
error recovery, and LangChain Expression Language (LCEL) composition. The notebook
implements all ten tasks in sequence. An OpenAI key is only needed for the live
model calls; prompt formatting, schema parsing, fallback behavior, and offline
checks can run without one.

## Project contents

```text
GenAI-Task25-PatanKhalilullaKhan/
├── README.md
├── requirements.txt
├── .gitignore
└── notebook/
    ├── GenAI_Assignment25_Prompting_Chains.ipynb
    └── build_assignment25.py
```

## Assignment tasks

1. Test `PromptTemplate` with different dynamic inputs.
2. Compare plain prompts with `ChatPromptTemplate` and explicit system/user messages.
3. Define a Pydantic response schema and parse structured output.
4. Validate malformed output, retry repair once when a model is available, and return a typed fallback if validation still fails.
5. Build a basic Prompt → LLM → OutputParser chain.
6. Route requests conditionally with `RunnableBranch`.
7. Generate an answer, summary, and follow-up questions concurrently with `RunnableParallel`, then merge the results.
8. Preserve and enrich inputs with `RunnablePassthrough`.
9. Run offline checks and an optional live model check.
10. Explain structured output, LCEL, and parallel versus conditional execution.

## Setup

Use Python 3.10 or later. From the project directory, create and activate a
virtual environment:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Set the OpenAI API key in PowerShell (do not commit keys to Git):

```powershell
$env:OPENAI_API_KEY = "your-key"
```

Alternatively, create a local `.env` file containing `OPENAI_API_KEY=your-key`.
The `.env` file is excluded by `.gitignore`.

## Run the notebook

Open `notebook/GenAI_Assignment25_Prompting_Chains.ipynb` in Jupyter or VS Code
and run the cells from top to bottom. Without an API key, the notebook skips live
model calls but still runs the prompt, parser, fallback, and evaluation examples.

## Rebuild the notebook

The notebook source is reproducible with the included standard-library script:

```powershell
python notebook\build_assignment25.py
```

## Notes

- The structured response has `answer`, `confidence` (0–1), and `source` fields.
- The retry handler catches output-format validation errors only. API and network
  failures remain visible instead of being presented as successful answers.
- Parallel branches increase aggregate model work and may cost more; use them
  when all independent outputs are needed.
