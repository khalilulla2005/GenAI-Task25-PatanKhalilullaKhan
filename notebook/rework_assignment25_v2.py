import nbformat as nbf
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableBranch, RunnableParallel, RunnableLambda
from pydantic import BaseModel, Field

# Setup Notebook
nb = nbf.v4.new_notebook()

# Markdown: Title
text_cell_1 = nbf.v4.new_markdown_cell("""# Assignment 25: Prompting & LangChain Chains (Production-Level Mastery)
**Student Name:** Patan Khalilulla Khan  
**Course:** GenAI Training Program  

---
## Assignment Overview
This notebook implements the production-level requirements for Assignment 25:
- **Pydantic Structured Output** (with source field)
- **RunnableBranch** for conditional routing
- **RunnableParallel** for concurrent LLM chains
- **LCEL-Based RAG** tested with diverse topics
""")

# Code: Setup
code_cell_1 = nbf.v4.new_code_cell("""# Setup
from langchain_core.documents import Document

# Load/Create diverse dataset
docs = [
    Document(page_content="The speed of light is approx 300,000 km/s.", metadata={"source": "physics"}),
    Document(page_content="LangChain is a framework for LLM apps.", metadata={"source": "tech"}),
]
vectorstore = FAISS.from_documents(docs, OpenAIEmbeddings())
retriever = vectorstore.as_retriever()
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

print("Vector store initialized with diverse topics.")
""")

# Markdown: Task 3/4
text_cell_2 = nbf.v4.new_markdown_cell("""## Task 3 & 4: Structured Output & Fallbacks
Implemented with Pydantic schema requiring 'source'.
""")

# Code: Task 3/4
code_cell_2 = nbf.v4.new_code_cell("""class Answer(BaseModel):
    answer: str = Field(description="The main answer")
    confidence: float = Field(description="Confidence score (0-1)")
    source: str = Field(description="The source of information")

parser = PydanticOutputParser(pydantic_object=Answer)
chain = (
    ChatPromptTemplate.from_template("Answer query: {q}\n{format}")
    | llm
    | parser
)

print(chain.invoke({"q": "What is AI?", "format": parser.get_format_instructions()}))
""")

# Markdown: Task 6
text_cell_3 = nbf.v4.new_markdown_cell("""## Task 6: Conditional Routing (RunnableBranch)
Route factual queries to retrieval and creative queries to direct LLM.
""")

# Code: Task 6
code_cell_3 = nbf.v4.new_code_cell("""# Branches
factual_chain = (retriever | (lambda x: f"Based on retrieved: {x[0].page_content}")) | llm
creative_chain = llm

branch = RunnableBranch(
    (lambda x: "fact" in x["q"].lower(), factual_chain),
    creative_chain
)

print("=== Task 6: Conditional Branch ===")
print(branch.invoke({"q": "What is a fact about light?"}))
print(branch.invoke({"q": "Write a poem about AI"}))
""")

# Markdown: Task 7
text_cell_4 = nbf.v4.new_markdown_cell("""## Task 7: RunnableParallel (Three Sub-Chains)
Executing Answer, Summary, and Follow-up chains in parallel.
""")

# Code: Task 7
code_cell_4 = nbf.v4.new_code_cell("""# Three distinct chains
chain_ans = ChatPromptTemplate.from_template("Answer: {q}") | llm | StrOutputParser()
chain_sum = ChatPromptTemplate.from_template("Summarize: {q}") | llm | StrOutputParser()
chain_fol = ChatPromptTemplate.from_template("Generate follow-up for: {q}") | llm | StrOutputParser()

parallel_chain = RunnableParallel({
    "answer": chain_ans,
    "summary": chain_sum,
    "follow_up": chain_fol
})

print("=== Task 7: Parallel Exec ===")
print(parallel_chain.invoke({"q": "LangChain basics"}))
""")

# Markdown: Task 9/10
text_cell_5 = nbf.v4.new_markdown_cell("""## Task 9 & 10: RAG Chain & Observations
Testing with diverse queries and answering observations.
""")

# Code: Task 9
code_cell_5 = nbf.v4.new_code_cell("""# Task 9: Executable LCEL RAG Chain
rag_chain = {"context": retriever, "q": RunnablePassthrough()} | ChatPromptTemplate.from_template("Ctx: {context}\nQ: {q}") | llm | StrOutputParser()

queries = ["What is the speed of light?", "What is LangChain?"]
for q in queries:
    print(f"Q: {q}\nA: {rag_chain.invoke(q)}")
""")

# Markdown: Task 10
text_cell_6 = nbf.v4.new_markdown_cell("""## Task 10: Observations & Insights
1. **Why structured output is important:** Enables programmatic reliability and prevents LLM hallucination in data extraction.
2. **Advantages of LCEL:** Provides first-class support for streaming, parallel execution, and built-in tracing/debugging.
3. **Parallel vs Conditional:** Parallel chains maximize throughput for independent operations, while conditional routing optimizes logic based on query semantics.
""")

nb['cells'] = [
    text_cell_1, code_cell_1, 
    text_cell_2, code_cell_2, 
    text_cell_3, code_cell_3, 
    text_cell_4, code_cell_4, 
    text_cell_5, code_cell_5,
    text_cell_6
]
nbf.write(nb, 'D:\\techtude\\GenAI-Task25-PatanKhalilullaKhan\\notebook\\GenAI_Assignment25_Prompting_Chains.ipynb')
print("Assignment 25 Reworked Notebook (Final Edition) generated!")
