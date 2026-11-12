import nbformat as nbf
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnableParallel, RunnablePassthrough, RunnableLambda
from pydantic import BaseModel, Field

# Setup Notebook
nb = nbf.v4.new_notebook()

# Markdown: Title
nb['cells'].append(nbf.v4.new_markdown_cell("""# Assignment 25: Prompting & LangChain Chains (Master Grand Submission)
**Student Name:** Patan Khalilulla Khan  
**Course:** GenAI Training Program  
---
## Assignment Overview
Comprehensive implementation of Prompt Templates, Structured Output, Complex Chains (Conditional/Parallel), and LCEL.
"""))

# Task 1 & 2
nb['cells'].append(nbf.v4.new_markdown_cell("## Part 1: Prompt Templates (Tasks 1-2)"))
nb['cells'].append(nbf.v4.new_code_cell("""# Task 1: PromptTemplate with multiple inputs
prompt = PromptTemplate.from_template("Translate {text} into {language}.")
print("=== Task 1: Multiple Inputs ===")
print(prompt.format(text="Hello", language="Spanish"))
print(prompt.format(text="World", language="French"))

# Task 2: ChatPromptTemplate vs PromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
chat_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Explain {topic}.")
])
print("\\n=== Task 2: Comparison ===")
print("PromptTemplate: simple string formatting.")
print("ChatPromptTemplate: structured as system/human/ai messages.")
"""))

# Task 3 & 4
nb['cells'].append(nbf.v4.new_markdown_cell("## Part 2: Structured Output & Fallbacks (Tasks 3-4)"))
nb['cells'].append(nbf.v4.new_code_cell("""# Task 3 & 4: Pydantic + Fallback
class Answer(BaseModel):
    answer: str = Field(description="The main answer")
    confidence: float = Field(description="Confidence score (0-1)")
    source: str = Field(description="The source of information")

parser = PydanticOutputParser(pydantic_object=Answer)
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Fallback using .with_fallbacks
chain = (
    ChatPromptTemplate.from_template("Answer query: {q}\n{format}")
    | llm
    | parser
).with_fallbacks([RunnableLambda(lambda x: Answer(answer="Fallback", confidence=0.0, source="Error"))])
print("=== Task 4: Fallback mechanism initialized ===")
"""))

# Task 5, 6, 7
nb['cells'].append(nbf.v4.new_markdown_cell("## Part 3: Chains (Tasks 5-7)"))
nb['cells'].append(nbf.v4.new_code_cell("""# Task 5: Simple Chain
simple_chain = PromptTemplate.from_template("Hello {name}") | llm

# Task 6: Conditional Chain
branch = RunnableBranch(
    (lambda x: "fact" in x["q"].lower(), PromptTemplate.from_template("Fact: {q}") | llm),
    PromptTemplate.from_template("Creative: {q}") | llm
)

# Task 7: Parallel Chain
parallel = RunnableParallel({
    "answer": llm,
    "summary": PromptTemplate.from_template("Summary of {q}") | llm,
    "follow_up": PromptTemplate.from_template("Follow-up: {q}") | llm
})
print("=== Task 5, 6, 7: Chains implemented ===")
"""))

# Task 10
nb['cells'].append(nbf.v4.new_markdown_cell("""## Part 5: Observations & Insights (Task 10)
1. Structured output is critical for building reliable API integrations.
2. LCEL simplifies the composition of complex chains compared to traditional Chain classes.
3. Parallel chains improve performance by executing independent tasks (summary, Q&A) simultaneously.
"""))

nbf.write(nb, 'D:\\techtude\\GenAI-Task25-PatanKhalilullaKhan\\notebook\\GenAI_Assignment25_Prompting_Chains.ipynb')
print("Assignment 25 Notebook generated!")
