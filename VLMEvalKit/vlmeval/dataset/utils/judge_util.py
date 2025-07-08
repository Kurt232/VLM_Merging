import os
from ...smp import load_env

INTERNAL = os.environ.get('INTERNAL', 0)


def build_judge(**kwargs):
    from ...api import SiliconFlowAPI
    model = kwargs.pop('model', None)
    kwargs.pop('nproc', None)
    load_env()
    # please set the key
    model = SiliconFlowAPI("deepseek-chat", key="", api_base="https://api.deepseek.com/v1/chat/completions", **kwargs)
    return model


DEBUG_MESSAGE = """
To debug the OpenAI API, you can try the following scripts in python:
```python
from vlmeval.api import OpenAIWrapper
model = OpenAIWrapper('gpt-4o', verbose=True)
msgs = [dict(type='text', value='Hello!')]
code, answer, resp = model.generate_inner(msgs)
print(code, answer, resp)
```
You cam see the specific error if the API call fails.
"""
