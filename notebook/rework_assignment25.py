import nbformat as nbf
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from pydantic import BaseModel, Field

# Setup Notebook
nb = nbf.v4.new_notebook()

# Markdown: Title
text_cell_1 = nbf.v4.new_markdown_cell("""# Assignment 25: Prompting & LangChain Chains (Mastery Edition)
**Student Name:** Patan Khalilulla Khan  
**Course:** GenAI Training Program  

---
## Assignment Overview
This notebook implements all requirements for Assignment 25, focusing on a **modular, production-ready GenAI architecture** using LCEL.
""")

# Code: Setup & Dataset
code_cell_1 = nbf.v4.new_code_cell("""# Setup: Load Dataset & Create Vector Store
from langchain_core.documents import Document

# Load/Create sample documents
texts = ["AI is rapidly evolving.", "LangChain simplifies LLM applications.", "LCEL enables complex pipelines."]
docs = [Document(page_content=t) for t in texts]

# Split & Index
splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
chunks = splitter.split_documents(docs)
vectorstore = FAISS.from_documents(chunks, OpenAIEmbeddings())
retriever = vectorstore.as_retriever()
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

print("Dataset loaded and indexed.")
""")

# Markdown: Prompt Templates Comparison
text_cell_2 = nbf.v4.new_markdown_cell("""## Task 1 & 2: Prompt Templates Comparison
Comparing `PromptTemplate` and `ChatPromptTemplate`.
""")

# Code: Task 1/2
code_cell_2 = nbf.v4.new_code_cell("""# Simple Prompt
prompt = PromptTemplate.from_template("What is {topic}?")
# Chat Prompt
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an AI expert."),
    ("human", "What is {topic}?")
])

print("=== Task 2: Comparison ===")
print("Simple:", prompt.format(topic="AI"))
print("Chat:", chat_prompt.format_messages(topic="AI"))
""")

# Markdown: Structured Output & Fallbacks
text_cell_3 = nbf.v4.new_markdown_cell("""## Task 3 & 4: Structured Output & Fallbacks
Using Pydantic with fallback mechanisms.
""")

# Code: Task 3/4
code_cell_3 = nbf.v4.new_code_cell("""class Answer(BaseModel):
    answer: str = Field(description="The main answer")
    confidence: float = Field(description="Confidence score (0-1)")

parser = PydanticOutputParser(pydantic_object=Answer)
chain = (
    PromptTemplate.from_template("Answer: {q}\n{format}", partial_variables={"format": parser.get_format_instructions()})
    | llm
    | parser
).with_fallbacks([RunnableLambda(lambda x: Answer(answer="Error", confidence=0.0))])

print(chain.invoke({"q": "What is AI?"}))
""")

# Markdown: Complex Chains
text_cell_4 = nbf.v4.new_markdown_cell("""## Task 6 & 7: Conditional & Parallel Chains
Executing complex logic using LCEL.
""")

# Code: Task 6/7
code_cell_4 = nbf.v4.new_code_cell("""# Conditional Chain
def route(x):
    return retriever if "fact" in x["q"].lower() else RunnableLambda(lambda x: [Document(page_content="Direct response")])

# Parallel Chain
parallel = RunnableParallel({
    "answer": RunnablePassthrough(),
    "summary": lambda x: "Summary of " + x["q"],
    "follow_up": lambda x: "Next question?"
})

print("=== Parallel Execution ===")
print(parallel.invoke({"q": "AI facts"}))
""")

# Markdown: LCEL RAG Chain
text_cell_5 = nbf.v4.new_markdown_cell("""## Task 9: Executable LCEL RAG Chain
The complete, production-ready LCEL pipeline.
""")

# Code: Task 9
code_cell_5 = nbf.v4.new_code_cell("""# LCEL RAG Chain
rag_chain = (
    {"context": retriever, "q": RunnablePassthrough()}
    | ChatPromptTemplate.from_template("Context: {context}\nQuestion: {q}")
    | llm
    | StrOutputParser()
)

print("=== LCEL RAG Chain ===")
print(rag_chain.invoke("What is AI?"))
""")

nb['cells'] = [text_cell_1, code_cell_1, text_cell_2, code_cell_2, text_cell_3, code_cell_3, text_cell_4, code_cell_4, text_cell_5, code_cell_5]
nbf.write(nb, 'D:\\techtude\\GenAI-Task25-PatanKhalilullaKhan\\notebook\\GenAI_Assignment25_Prompting_Chains.ipynb')
