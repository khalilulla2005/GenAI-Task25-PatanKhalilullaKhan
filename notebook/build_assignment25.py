import json
from pathlib import Path


def markdown(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


cells = [
    markdown(
        """# Assignment 25: Prompting and LangChain Chains

**Student:** Patan Khalilulla Khan

**Course:** GenAI Training Program

This notebook builds ten LangChain examples in order, from prompt formatting and
Pydantic parsing through LCEL composition, conditional routing, and parallel work.
The notebook does not require retrieval or a vector database.
"""
    ),
    markdown(
        """## Setup

Install the packages listed in `requirements.txt`. For live model calls, set
`OPENAI_API_KEY` in your environment (or a local `.env` file). Tasks 1-4 and the
offline checks in Task 9 can be run without a key; model-backed tasks clearly skip
their live invocation when no key is configured.
"""
    ),
    code(
        """import os

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.prompts.chat import (
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_openai import ChatOpenAI

load_dotenv()
api_key_configured = bool(os.getenv("OPENAI_API_KEY"))
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0) if api_key_configured else None

print("OpenAI model is ready." if llm else
      "OPENAI_API_KEY is not set. Prompt/parser examples will run; live model calls will be skipped.")
"""
    ),
    markdown(
        """## Task 1 — Test `PromptTemplate` with dynamic inputs

`PromptTemplate` formats a plain-text prompt. The same template can safely be reused
with different input values.
"""
    ),
    code(
        """translation_prompt = PromptTemplate.from_template(
    "Translate the following text into {language}: {text}"
)

task1_examples = [
    {"language": "Spanish", "text": "Good morning"},
    {"language": "French", "text": "Thank you"},
]

for values in task1_examples:
    print(translation_prompt.format(**values))

assert translation_prompt.input_variables == ["language", "text"]
"""
    ),
    markdown(
        """## Task 2 — Build a `ChatPromptTemplate` and compare prompt types

The chat template below uses explicit `SystemMessage` and `HumanMessage` objects.
This keeps the assistant's role separate from the user's changing input.
"""
    ),
    code(
        """chat_prompt = ChatPromptTemplate.from_messages(
    [
                SystemMessagePromptTemplate.from_template(
                    "You are a clear, helpful teaching assistant."
                ),
                HumanMessagePromptTemplate.from_template(
                    "Explain {topic} in one short paragraph."
                ),
    ]
)
formatted_messages = chat_prompt.format_messages(topic="large language models")

print("ChatPromptTemplate messages:")
for message in formatted_messages:
    print(f"{message.type}: {message.content}")

plain_prompt = PromptTemplate.from_template("Explain {topic} in one short paragraph.")
print("\\nPromptTemplate result:")
print(plain_prompt.format(topic="large language models"))
print("\\nComparison: PromptTemplate formats one text string; ChatPromptTemplate "
      "formats a role-labeled sequence of messages.")

assert isinstance(formatted_messages[0], SystemMessage)
assert isinstance(formatted_messages[1], HumanMessage)
"""
    ),
    markdown(
        """## Task 3 — Define a Pydantic schema and structured output parser

The schema validates the answer's shape and confidence range. The parser turns
JSON-like model text into a typed Python object and supplies format instructions
that can be added to the prompt.
"""
    ),
    code(
        """class StructuredAnswer(BaseModel):
    answer: str = Field(min_length=1, description="A concise answer to the question")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence from 0 to 1")
    source: str = Field(min_length=1, description="Source or basis for the answer")


answer_parser = PydanticOutputParser(pydantic_object=StructuredAnswer)
structured_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content="Answer accurately and follow the requested output schema."),
        ("human", "Question: {question}\\n\\n{format_instructions}"),
    ]
).partial(format_instructions=answer_parser.get_format_instructions())

valid_example = (
    '{"answer": "LCEL composes LangChain runnables into a pipeline.", '
    '"confidence": 0.94, "source": "LangChain concept"}'
)
parsed_example = answer_parser.parse(valid_example)
print(parsed_example.model_dump())
assert isinstance(parsed_example, StructuredAnswer)
"""
    ),
    markdown(
        """## Task 4 — Handle invalid output with validation, one repair retry, and fallback

Parsing errors are handled explicitly. If a repair model is supplied, malformed
output is sent back for one schema-guided correction; if parsing still fails (or no
repair model is available), a typed fallback result is returned. Model/network
errors are intentionally not swallowed.
"""
    ),
    code(
        """repair_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content="Repair the supplied response so it matches the JSON schema exactly. "
            "Return only the corrected structured response."
        ),
        (
            "human",
            "Question: {question}\\n"
            "Required format:\\n{format_instructions}\\n"
            "Invalid response:\\n{invalid_output}",
        ),
    ]
)


def parse_with_retry(raw_output, question, repair_model=None, max_retries=1):
    if max_retries < 0:
        raise ValueError("max_retries must be zero or greater")

    attempt = 0
    while True:
        try:
            return answer_parser.parse(raw_output)
        except OutputParserException:
            if repair_model is None or attempt >= max_retries:
                return StructuredAnswer(
                    answer="The response could not be validated. Please try again.",
                    confidence=0.0,
                    source="Fallback: structured output validation failed",
                )

            raw_output = (
                repair_prompt | repair_model | StrOutputParser()
            ).invoke(
                {
                    "question": question,
                    "format_instructions": answer_parser.get_format_instructions(),
                    "invalid_output": raw_output,
                }
            )
            attempt += 1


invalid_example = "This is not valid JSON"
fallback_result = parse_with_retry(invalid_example, "What is LCEL?")
print(fallback_result.model_dump())
assert fallback_result.confidence == 0.0
assert fallback_result.source.startswith("Fallback:")
"""
    ),
    markdown(
        """## Task 5 — Create a standalone basic chain

This is the basic LCEL sequence: **Prompt → LLM → OutputParser**. The output parser
returns plain text instead of a provider-specific message object.
"""
    ),
    code(
        """basic_prompt = PromptTemplate.from_template(
    "Explain {topic} to a beginner in two sentences."
)
basic_chain = basic_prompt | llm | StrOutputParser() if llm else None

if basic_chain:
    print(basic_chain.invoke({"topic": "LangChain Expression Language (LCEL)"}))
else:
    print("Skipped live chain invocation; set OPENAI_API_KEY to run Task 5.")
"""
    ),
    markdown(
        """## Task 6 — Route requests conditionally with `RunnableBranch`

Conditional routing is useful when one input should select exactly one path. Here
the requested response style selects either a creative or a direct-answer chain.
"""
    ),
    code(
        """creative_chain = (
    ChatPromptTemplate.from_messages(
        [
            SystemMessage(content="Answer in a playful, imaginative style."),
            ("human", "{question}"),
        ]
    )
    | llm
    | StrOutputParser()
) if llm else None

direct_chain = (
    ChatPromptTemplate.from_messages(
        [
            SystemMessage(content="Answer directly and factually."),
            ("human", "{question}"),
        ]
    )
    | llm
    | StrOutputParser()
) if llm else None

style_router = (
    RunnableBranch(
        (lambda values: values["style"] == "creative", creative_chain),
        direct_chain,
    )
    if llm
    else None
)

if style_router:
    print(
        style_router.invoke(
            {"style": "direct", "question": "What does a prompt template do?"}
        )
    )
else:
    print("Skipped live branch invocation; set OPENAI_API_KEY to run Task 6.")
"""
    ),
    markdown(
        """## Task 7 — Run answer, summary, and follow-up chains in parallel

`RunnableParallel` sends the same input to three independent chains and merges
their results into one dictionary. A final runnable formats that dictionary as a
single report.
"""
    ),
    code(
        """if llm:
    answer_branch = (
        ChatPromptTemplate.from_template("Answer this question clearly: {question}")
        | llm
        | StrOutputParser()
    )
    summary_branch = (
        ChatPromptTemplate.from_template(
            "Give a one-sentence summary of the key idea in this question: {question}"
        )
        | llm
        | StrOutputParser()
    )
    follow_up_branch = (
        ChatPromptTemplate.from_template(
            "Suggest two useful follow-up questions for: {question}"
        )
        | llm
        | StrOutputParser()
    )

    parallel_chain = RunnableParallel(
        answer=answer_branch,
        summary=summary_branch,
        follow_up_questions=follow_up_branch,
    )
    parallel_and_merge = parallel_chain | RunnableLambda(
        lambda result: {
            **result,
            "merged_output": (
                f"Answer: {result['answer']}\\n\\n"
                f"Summary: {result['summary']}\\n\\n"
                f"Follow-up questions: {result['follow_up_questions']}"
            ),
        }
    )
    parallel_result = parallel_and_merge.invoke(
        {"question": "How does structured output help an LLM application?"}
    )
    print("Merged output keys:", list(parallel_result))
    print(parallel_result["merged_output"])
else:
    parallel_chain = None
    parallel_and_merge = None
    print("Skipped live parallel invocation; set OPENAI_API_KEY to run Task 7.")
"""
    ),
    markdown(
        """## Task 8 — Preserve and enrich input with `RunnablePassthrough`

`RunnablePassthrough.assign` keeps the original fields while adding derived values.
This is useful in LCEL pipelines when downstream steps need both the original input
and computed context.
"""
    ),
    code(
        """prepare_question = RunnablePassthrough.assign(
    normalized_question=RunnableLambda(
        lambda values: values["question"].strip()
    ),
    question_length=RunnableLambda(
        lambda values: len(values["question"].strip())
    ),
)
prepared = prepare_question.invoke(
    {"question": "  Why compose chains with LCEL?  ", "user_id": "demo"}
)
print(prepared)
assert prepared["user_id"] == "demo"
assert prepared["normalized_question"] == "Why compose chains with LCEL?"

enriched_prompt = ChatPromptTemplate.from_template(
    "Answer this question in one sentence ({question_length} characters): "
    "{normalized_question}"
)
enriched_chain = (
    prepare_question | enriched_prompt | llm | StrOutputParser()
) if llm else None
if enriched_chain:
    print(enriched_chain.invoke({"question": "Why compose chains with LCEL?"}))
else:
    print("Skipped live enriched-chain invocation; set OPENAI_API_KEY to run it.")
"""
    ),
    markdown(
        """## Task 9 — Evaluate the parser and chain components

The offline assertions verify template inputs, typed parsing, confidence validation,
and deterministic fallback behavior. Live model checks run only when an API key is
configured.
"""
    ),
    code(
        """# Valid structured data parses into the expected typed shape.
checked = answer_parser.parse(valid_example)
assert checked.answer
assert 0.0 <= checked.confidence <= 1.0
assert checked.source

# An out-of-range confidence is rejected by Pydantic through the parser.
invalid_confidence = (
    '{"answer": "Example", "confidence": 1.5, "source": "Test"}'
)
checked_fallback = parse_with_retry(invalid_confidence, "Test validation")
assert checked_fallback.confidence == 0.0

if llm:
    raw_structured = (structured_prompt | llm | StrOutputParser()).invoke(
        {"question": "What is the purpose of Pydantic output parsing?"}
    )
    live_result = parse_with_retry(
        raw_structured,
        "What is the purpose of Pydantic output parsing?",
        repair_model=llm,
        max_retries=1,
    )
    print(live_result.model_dump())
else:
    print("Offline parser checks passed. Live model evaluation was skipped.")

print("Task 9 checks passed.")
"""
    ),
    markdown(
        """## Task 10 — Written answers

### 1. Why use structured output?

Structured output converts free-form model text into a predictable, validated
shape. A Pydantic schema catches missing fields and invalid values, making the
result safer to pass to application code, APIs, and databases.

### 2. What are the benefits of LCEL?

LCEL composes prompts, models, parsers, and runnable transformations with a
consistent pipe syntax. Components remain reusable and testable, while the
composition supports streaming, batching, and asynchronous execution where each
component allows it.

### 3. When should parallel execution be used instead of conditional routing?

Use parallel execution when several independent results are all needed for the
same input, such as an answer, summary, and follow-up questions. Use conditional
routing when a condition should choose one path, such as selecting a response
style or model. Parallel work can reduce wall-clock time for independent tasks,
but it may increase total token use and cost; routing avoids work on unused paths.
"""
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.10+"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

output_path = Path(__file__).with_name("GenAI_Assignment25_Prompting_Chains.ipynb")
output_path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"Wrote {len(cells)} cells to {output_path}")
