import argparse
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel
from pydantic import Field

from utils import setting_env_vars

def main(args):
    
  setting_env_vars()

  llm = ChatOllama(
    model='llama3.1',
    temperature=0.0)
  
  tagging_prompt = ChatPromptTemplate.from_template(
    template="""
Extract the desired information from the following passage.

Only extract the properties mentioned in the 'Classification' function.

Passage:
{input}
"""
  )

  structured_llm = llm.with_structured_output(Classification)

  # スペイン語
  inp = 'Estoy increiblemente contento de haberte conocido! Creo que seremos muy buenos amigos!'
  prompt = tagging_prompt.invoke({'input': inp})
  response = structured_llm.invoke(prompt)
  print(response)

  structured_llm_rev1 = llm.with_structured_output(ClassificationRev1)
  response = structured_llm_rev1.invoke(prompt)
  print(response)
  
  # スペイン語
  inp = 'Estoy muy enojado con vos! Te voy a dar tu merecido!'
  prompt = tagging_prompt.invoke({"input": inp})
  response = structured_llm_rev1.invoke(prompt)
  print(response)

  # 英語
  inp = 'Weather is ok here, I can go outside without much more than a coat'
  prompt = tagging_prompt.invoke({"input": inp})
  response = structured_llm_rev1.invoke(prompt)
  print(response)

class Classification(BaseModel):
  sentiment: str = Field(description='The sentiment of the text')
  aggressiveness: int = Field(
    description='How aggressive the text is on a scale from 1 to 10'
  )
  language: str = Field(description='The language the text is written in')

class ClassificationRev1(BaseModel):
  sentiment: str = Field(
    enum=[
      'happy',
      'neutral',
      'sad'
    ],
    description='The sentiment of the text')
  aggressiveness: int = Field(
    enum=[
      1,
      2,
      3,
      4,
      5
    ],
    description='describes how aggressive the statement is, the higher the number the more aggressive'
  )
  language: str = Field(
    enum=[
      'spanish',
      'english',
      'french',
      'german',
      'italian'
    ],
    description='The language the text is written in')

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  args = parser.parse_args()
  main(args=args)
