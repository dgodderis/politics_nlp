import yaml
import os

from langchain.llms import OpenAI


with open("config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
os.environ["OPENAI_API_KEY"] = config['OPENAI_API_KEY']

llm = OpenAI(temperature=0.9, model_name='text-davinci-003')

text = "What would be a good company name for a company that makes colorful socks?"
print(llm(text))
