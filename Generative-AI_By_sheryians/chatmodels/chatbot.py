from dotenv import load_dotenv

load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage

model = ChatMistralAI(model="mistral-small-2506",temperature=0.9)


print("choose your AI Mode")
print("Press 1 for angry mode")
print("Press 2 for funny mode")
print("Press 3 for sad mode")

choice = int(input("tell your response:- "))
mode = 'normal'

if choice == 1:
    mode = 'angry'
elif choice == 2:
    mode = 'funny'
elif choice == 3:
    mode = 'sad'



messages = [
    SystemMessage(content=f"you are a {mode} AI agent")
]

print("--------------- Welcome type 0 to exit the application ---------")

while True:
    prompt = input("You : ")
    messages.append(HumanMessage(content=prompt))

    if prompt == '0':
        break

    response = model.invoke(messages)
    messages.append(AIMessage(content=response.content))
    print("Bot : ", response.content)