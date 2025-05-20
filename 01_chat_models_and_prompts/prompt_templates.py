import argparse
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

from utils import setting_env_vars

def main(args):
  
  setting_env_vars()

  model = ChatOllama(
    model='llama3.1',
    temperature=0)
  
  system_template = 'Translate the following from English into {language}'

  prompt_template = ChatPromptTemplate.from_messages(
    [
      ('system', system_template),
      ('user', '{text}')
    ]
  )

  prompt = prompt_template.invoke({
    'language': 'Italian',
    'text': 'hi!'
  })

  response = model.invoke(prompt)
  print(response.content)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  args = parser.parse_args()
  main(args=args)