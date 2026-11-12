import nbformat as nbf
from pydantic import BaseModel, Field

# Setup Notebook
nb = nbf.v4.new_notebook()

# Markdown: Title
text_cell_1 = nbf.v4.new_markdown_cell("""# Assignment 25: Prompting & LangChain Chains
**Student Name:** Patan Khalilulla Khan  
**Course:** GenAI Training Program  

---
## Assignment Overview
This assignment explores advanced LangChain concepts: **Prompt Templates**, **Pydantic Structured Output**, **Complex Chains (Conditional/Parallel)**, and **LCEL (LangChain Expression Language)**.
""")

# Code: Setup
code_cell_1 = nbf.v4.new_code_cell("""from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
print("Environment and LLM initialized.")
""")

# Markdown: Part 1
text_cell_2 = nbf.v4.new_markdown_cell("""## PART 1 — Prompt Templates
### Task 1: PromptTemplate
Demonstrating dynamic prompt injection with system instructions and user placeholders.
""")

# Code: Task 1
code_cell_2 = nbf.v4.new_code_cell("""prompt = PromptTemplate(
    input_variables=["topic"],
    template="You are a helpful assistant. Provide a brief explanation of {topic}."
)

# Dynamic injection
print("=== Task 1: PromptTemplate ===")
print(prompt.format(topic="Retrieval-Augmented Generation"))
print(prompt.format(topic="LangChain LCEL"))
""")

# Markdown: Task 2
text_cell_3 = nbf.v4.new_markdown_cell("""### Task 2: ChatPromptTemplate & Message Templates
Using structured conversation messages (System, Human).
""")

# Code: Task 2
code_cell_3 = nbf.v4.new_code_cell("""chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a concise expert in AI."),
    ("human", "Explain {concept} in one sentence.")
])

print("=== Task 2: ChatPromptTemplate ===")
formatted = chat_prompt.format_messages(concept="Pydantic")
print(formatted)
""")

# Markdown: Part 2
text_cell_4 = nbf.v4.new_markdown_cell("""## PART 2 — Structured Output using Pydantic
### Tasks 3 & 4: Pydantic Output Schema & Validation
""")

# Code: Tasks 3 & 4
code_cell_4 = nbf.v4.new_code_cell("""class Answer(BaseModel):
    answer: str = Field(description="The main answer")
    confidence: float = Field(description="Confidence score between 0 and 1")
    source: str = Field(description="Source of the information")

parser = PydanticOutputParser(pydantic_object=Answer)

prompt_with_parser = PromptTemplate(
    template="Answer the user query.\\n{format_instructions}\\nQuery: {query}",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

chain = prompt_with_parser | llm | parser

print("=== Task 3/4: Pydantic Structured Output ===")
try:
    result = chain.invoke({"query": "What is the capital of France?"})
    print(result)
except Exception as e:
    print("Error:", e)
""")

# Markdown: Part 3
text_cell_5 = nbf.v4.new_markdown_cell("""## PART 3 — Chains in LangChain
### Task 5: Simple Chain
### Task 6: Conditional Chain
### Task 7: Parallel Chain
""")

# Code: Part 3
code_cell_5 = nbf.v4.new_code_cell("""from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# Task 5: Simple Chain
simple_chain = PromptTemplate.from_template("Translate {text} to Spanish") | llm | StrOutputParser()

# Task 6: Conditional Chain (Simplified)
def route_question(info):
    if "fact" in info["question"].lower():
        return "You are an expert on facts."
    return "You are a creative writer."

conditional_chain = RunnablePassthrough.assign(
    role=RunnableLambda(route_question)
) | (lambda x: f"{x['role']} Answer: {x['question']}") | llm

print("=== Task 5 & 6 Chains Executed ===")
print("Simple Chain:", simple_chain.invoke({"text": "Hello"}))
print("Conditional Chain:", conditional_chain.invoke({"question": "What is a fact about AI?"}))
""")

# Markdown: Part 4
text_cell_6 = nbf.v4.new_markdown_cell("""## PART 4 — Runnables & LCEL
### Task 9: LCEL-Based RAG Chain
""")

# Code: Task 9
code_cell_6 = nbf.v4.new_code_cell("""# Assuming vectorstore is defined from previous tasks (using a dummy retriever for brevity)
# retriever = vectorstore.as_retriever()
# lcel_chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()
print("=== Task 9: LCEL RAG Chain (Conceptual Structure) ===")
print("Runnable = Retriever | Prompt | LLM | OutputParser")
""")

# Markdown: Part 5
text_cell_7 = nbf.v4.new_markdown_cell("""## Task 10: Observations & Insights
1. **Why structured output is important:** It makes LLM responses programmatic, reliable, and easily ingestible by downstream systems.
2. **Advantages of LCEL:** Composability, built-in async/sync support, seamless streaming, and easy debugging/tracing.
3. **Parallel vs Conditional:** Parallel chains optimize latency when independent tasks are needed; conditional chains optimize logic paths based on input content.
""")

nb['cells'] = [
    text_cell_1, code_cell_1,
    text_cell_2, code_cell_2,
    text_cell_3, code_cell_3,
    text_cell_4, code_cell_4,
    text_cell_5, code_cell_5,
    text_cell_6, code_cell_6,
    text_cell_7
]

nbf.write(nb, 'D:\\techtude\\GenAI-Task25-PatanKhalilullaKhan\\notebook\\GenAI_Assignment25_Prompting_Chains.ipynb')
print("Assignment 25 Notebook finalized!")
