from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Local LLM
llm = Ollama(
    model="phi3",
    temperature=0.3
)

prompt = ChatPromptTemplate.from_template("""
The system predicted the handwritten digit {digit} with confidence {confidence}.

Explain this in 3 short sentences:
- Focus only on the visible shape and strokes of the digit
- Be simple and user-friendly
- Do NOT mention neural networks, training data, or AI concepts
- Do NOT ask questions
""")

parser = StrOutputParser()

chain = prompt | llm | parser

def explain_prediction(digit: int, confidence: float) -> str:
    return chain.invoke({
        "digit": digit,
        "confidence": f"{confidence:.2f}"
    })