import getpass
import os

def setting_env_vars():
  try:
    from dotenv import load_dotenv
    load_dotenv()
  except ImportError:
    pass

  os.environ['LANGSMITH_TRACING'] = 'true'
  if 'LANGSMITH_API_KEY' not in os.environ:
    os.environ['LANGSMITH_API_KEY'] = getpass.getpass(
      prompt='Enter your LangSmith API key (optional): '
    )
  if 'LANGSMITH_PROJECT' not in os.environ:
    os.environ['LANGSMITH_PROJECT'] = getpass.getpass(
      prompt='Enter your LangSmith Project Name (default = "default"):'
    )
    if not os.environ.get('LANGSMITH_PROJECT'):
      os.environ['LANGSMITH_PROJECT'] = 'default'
  
  print('LANGSMITH_TRACING:{}'.format(os.environ['LANGSMITH_TRACING']))
  print('LANGSMITH_API_KEY:{}...{}'.format(os.environ['LANGSMITH_API_KEY'][:4], os.environ['LANGSMITH_API_KEY'][-4:]))
  print('LANGSMITH_PROJECT:{}'.format(os.environ['LANGSMITH_PROJECT']))
