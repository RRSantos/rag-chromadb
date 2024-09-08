from langchain.prompts import ChatPromptTemplate

from langchain_core.language_models.chat_models import BaseChatModel


# from langchain_groq import ChatGroq

from database import Database


PROMPT_TEMPLATE = """
Responda à pergunta com base apenas no contexto abaixo:

{context}

---

Você deverá ser cordial, chamar a pessoa pelo nome {name} e responder sempre em português.
Se você não souber a resposta, responda 'Sinto muito. Eu não sei a resposta paras isso'.

---
Responda à seguinte pergunta com base no contexto acima: {question}

"""


def query_rag(query_text: str, user_name: str, database: Database, chat: BaseChatModel):

    results = database._db.similarity_search_with_score(query_text, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(
        context=context_text, name=user_name, question=query_text
    )

    response_text = chat.invoke(prompt)

    return response_text
