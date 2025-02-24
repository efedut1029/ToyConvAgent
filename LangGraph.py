# advanced

#pip install -U tavily-python langchain_community
from typing import Annotated
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict
import gradio as gr
import os
import openai


from IPython.display import Image, display

from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

MODEL = "gpt-4o"

from dotenv import load_dotenv

load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'your-key-if-not-using-env')
os.environ['TAVILY_API_KEY'] = os.getenv('TAVILY_API_KEY')

class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)


tool = TavilySearchResults(max_results=2)
tools = [tool]
llm = ChatOpenAI(temperature=0.7, model_name=MODEL)
llm_with_tools = llm.bind_tools(tools)

def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
# Any time a tool is called, we return to the chatbot to decide the next step
graph_builder.add_edge("tools", "chatbot")
graph_builder.set_entry_point("chatbot")
graph = graph_builder.compile()

try:
     display(Image(graph.get_graph().draw_mermaid_png()))
except Exception:
     # This requires some extra dependencies and is optional
     pass



# Function to generate an AI image using DALL·E 3
def generate_personality_image(description):
    """Generate an image using DALL·E 3 based on the personality description and return the URL."""
    
    image_prompt = f"A unique, highly detailed, cinematic digital artwork representation of a chatbot personality described as: {description}. \
                    The image should reflect its style, attitude, and essence."
    
    response = openai.images.generate(
        model="dall-e-3",
        prompt=image_prompt,
        size="1024x1024",
        n=1
    )

    return response.data[0].url  # Return the image URL

# Function to generate a detailed personality prompt using GPT-4o
def generate_personality_prompt(description):
    """Use GPT-4o to generate a detailed system prompt from a short user description."""
    
    prompt_template = f"""
    You are an AI that specializes in creating personalities for chatbots.
    Based on the following user request, generate a detailed system prompt 
    that defines how the chatbot should respond.

    User Request: "{description}"
    
    The prompt should include:
    - The chatbot’s personality traits.
    - The tone and style of responses.
    - Any unique quirks or mannerisms.
    - A brief explanation of how the chatbot views the world.

    Generate the system prompt below:
    """

    response = openai.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a personality creator for AI agents."},
            {"role": "user", "content": prompt_template}
        ]
    )

    return response.choices[0].message.content


def chat_with_graph(message, history, personality_prompt):
    """Processes user messages and allows real-time personality changes."""

    # Check if user wants to change personality
    if message.startswith("/set_personality"):
        new_description = message.replace("/set_personality", "").strip()
        if new_description:
            personality_prompt = generate_personality_prompt(new_description)
            image_url = generate_personality_image(new_description)  # Generate image
            
            print(f"✅ Personality updated to: {new_description}")
            print(f"🖼️ View Image: {image_url}")  # Display image URL
            
            return f"✅ Personality updated to: {new_description}", history, personality_prompt, image_url
        else:
            return "⚠ Please provide a new personality description after /set_personality.", history, personality_prompt, None

    # System message always includes the latest personality
    system_message = {"role": "system", "content": personality_prompt}

    # Convert history into LangGraph-compatible format
    formatted_history = [{"role": "user", "content": msg} if i % 2 == 0 
                         else {"role": "assistant", "content": response}
                         for i, (msg, response) in enumerate(history)]
    
    # Include system message, conversation history, and new user message
    messages = [system_message] + formatted_history + [{"role": "user", "content": message}]

    res = []

    for event in graph.stream({"messages": messages}):
        for value in event.values():
            assistant_response = value["messages"][-1].content  # Extract last assistant response
            res.append(assistant_response)

    response_text = res[-1] if res else "Sorry, I didn't understand."

    # Store conversation history correctly
    history.append((message, response_text))

    return response_text, history, personality_prompt, None  # No image for normal messages



if __name__ == "__main__":
    personality_description = input("Describe the chatbot’s personality (e.g., 'Einstein with Batman's attitude'): ")
    personality_prompt = generate_personality_prompt(personality_description)
    image_url = generate_personality_image(personality_description)  # Generate initial image

    print(f"🖼️ View Personality Image: {image_url}")  

    chat_history = []
    
    while True:
        query = input("Ask a question (or type 'exit/quit/q' to quit): ")
        if query.lower() in ["exit", "quit", "q"]:
            break

        response, chat_history, personality_prompt, new_image_url = chat_with_graph(query, chat_history, personality_prompt)

        print("\n🤖", response)
        
        if new_image_url:
            print(f"🖼️ Updated Personality Image: {new_image_url}")  # Show new image URL when personality changes


#gr.ChatInterface(fn=chat_with_graph, title="General Agent with Dynamic Personality").launch(share=True)



