import argparse
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage

from utils import setting_env_vars

def main(args):
  
  setting_env_vars()

  llm = ChatOllama(
    model='llama3.1',
    temperature=0)
  
  messages = [
    SystemMessage('Translate the following from English into Italian'),
    HumanMessage('hi!')
  ]
  ai_msg = llm.invoke(messages)
  print(ai_msg)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  args = parser.parse_args()
  main(args=args)