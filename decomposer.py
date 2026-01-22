from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
import re
from langchain_core.output_parsers.openai_tools import PydanticToolsParser

PATH_RE = r"<CHAIN>(.*)<\/CHAIN>"
def parse_sub_questions(prediction):
    sub_questions = []
    path = re.search(PATH_RE, prediction)
    if path is None:
        return sub_questions
    path = path.group(1)
    path = path.split("<SEP>")
    if len(path) == 0:
        return sub_questions
    for sub_question in path:
        sub_question = sub_question.strip()
        if sub_question == "":
            continue
        sub_questions.append(sub_question)
    return sub_questions

def question_decompose(llm, question):
    system = """You are an expert at converting user questions into sub reasoning questions.

    Perform question decomposition. Given a user question, break it down into one sub question or several reasonable distinct sub questions that 
    you need to answer in order to answer the original question.
    Each sub question is a step to form the reasoning chain to get the final answer.
    Please break the original question into as few sub-questions as possible.

    The second question should use the answer from the first question but not explicitly mention it.
    Instead, it should refer to it indirectly.
    answer with the format:<CHAIN>sub_question1<SEP>sub_question2</CHAIN>
    Here is example:
    Example1:Converting the original question to one sub question.
    Question:what does jamaican people speak?
    Answers:<CHAIN>what does jamaican people speak?</CHAIN>

    Example2:Converting the original question to two sub question.
    Question:Who did George W. Bush run against for the second term?
    Answers:
    <CHAIN>In which election did George W. Bush seek to remain in office?<SEP>Who was the main challenger in that political contest?</CHAIN>

    Example3:Converting the original question to three sub question.
    Question:Who is the grandfather of the wife of The 44th President of the United States?
    Answers:<CHAIN>Who is the 44th President of the United States?<SEP>Who is the wife of that person?<SEP>Who is the grandfather of that woman?</CHAIN>
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "{question}"),
        ]
    )

    chain = prompt | llm
    answer = chain.invoke(
        {
            "input_language": "English",
            "output_language": "English",
            "question": question,
        }
    )
    sub_questions = parse_sub_questions(answer.content)
    return sub_questions

