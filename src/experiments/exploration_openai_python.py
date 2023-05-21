import yaml
import os

from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI

llm = OpenAI(temperature=0.9)
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType

# Return messages on a topic
with open("config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

os.environ["OPENAI_API_KEY"] = config['OPENAI_API_KEY']
os.environ["GOOGLE_API_KEY"] = config['GOOGLE_API_KEY']
os.environ["GOOGLE_CSE_ID"] = config['GOOGLE_CSE_ID']

chat = ChatOpenAI(temperature=0.25, model_name='gpt-3.5-turbo')

messages = [
    SystemMessage(content="Jij bent een hulpvaardige chatbot dat politieke verslaggeving doet."),
    HumanMessage(content="Geef het standpunt van de CD&V wat betreft kernenergie. Vat samen in enkele zinnen.")
]
chat(messages)

# Experiment with google requests
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.tools import Tool
from langchain.agents import load_tools
tools = load_tools(["google-search"])

search = GoogleSearchAPIWrapper()

tool = Tool(
    name = "Google Search",
    description="Search Google for recent results.",
    func=search.run
)
tool.run("How old was Obama in 2021?")

# Use tools with agents
llm = OpenAI(temperature=0, model_name='gpt-3.5-turbo')
# llm = OpenAI(temperature=0, model_name='text-davinci-003')

tools = load_tools(["google-search"], llm=llm)
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
agent.run("Geef 5 Vlaamse of federale partijleden van de CD&V in 2023.")
# Final Answer: Tonia Antoniazzi, Karin Brouwers, Yves Leterme, Manfred Weber, and Cindy Franssen.
agent.run("Give 5 Flemish or federal party members of the Belgian political party CD&V.")
# Final Answer: Wouter Beke, Joachim Coens, Liesbeth Homans, Bart Tommelein, Geert Bourgeois.
agent.run("Wat is het standpunt van Wouter beke wat betreft uitstap kernenergie. Geef de originele tekst terug in 5 zinnen.")
# Final Answer: Wouter Beke is in favor of nuclear energy, but wants to reduce Belgium's dependence on foreign energy by closing its seven nuclear plants by 2025.
agent.run("Wat is het standpunt van Tine Vanderstraeten wat betreft uitstap kernenergie. Geef de originele tekst terug in 5 zinnen.")
# Final Answer: 'Minister Tinne Van der Straeten (Groen) is voorstander van een kernuitstap tegen 2025. Ze heeft aangegeven dat de kerncentrales in 2025 gesloten zullen worden en dat de energiefactuur zal dalen. Daarnaast heeft ze aangegeven dat er een akkoord is met Engie over de verlenging van twee kerncentrales en dat de ontmanteling van de kerncentrales en de berging van nucleair afval door een externe partij gedaan zal worden.'
agent.run("Wat is het standpunt van enkele partijleden van de Partij van de Arbeid wat betreft uitstap kernenergie? Dit dient enkel te gebeuren voor de partij PVDA. Geef dit terug als opsomming. Geef voor elk partijlid de naam van het lid en zijn of haar mening in enkele zinnen. Bv. 'Raoul Hedebouw: België moet onafhankelijk zijn van kernenergie tegen 2025.")
# Dramatisch

agent.run("Wat is het standpunt van de partij Vooruit wat betreft kernuitstap? Belangrijk is om dit in minstens 10 zinnen terug te geven. Denk stap per stap. Denk goed na of het antwoord ook minstens 10 zinnen bevat.")
