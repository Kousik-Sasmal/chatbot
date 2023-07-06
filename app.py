import streamlit as st
import openai
from streamlit_chat import message
from utils import get_conversation_string, find_match, query_refiner

from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)

llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key="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")


st.title("Chatbot of INDIAN Budget(23-24)")


if 'responses' not in st.session_state:
    st.session_state['responses'] = ["How can I assist you?"]

if 'requests' not in st.session_state:
    st.session_state['requests'] = []

if 'buffer_memory' not in st.session_state:
    st.session_state.buffer_memory=ConversationBufferWindowMemory(k=3,return_messages=True)


system_msg_template = SystemMessagePromptTemplate.from_template(template="""Answer the question as truthfully 
as possible using the provided context, and if the answer is not contained within the text below, say 'I don't know'""")


human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")

prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), 
                                                    human_msg_template])

conversation = ConversationChain(memory=st.session_state.buffer_memory, prompt=prompt_template, 
                                 llm=llm, verbose=True)



textcontainer = st.container()
response_container = st.container()

try:
    with textcontainer:
        query = st.text_input("Query: ", key="input")
        if query:
            with st.spinner("typing..."):
                conversation_string = get_conversation_string()
                # st.code(conversation_string)

                refined_query = query_refiner(conversation_log=conversation_string,query=query)
                st.subheader('Refined Query:')
                st.write(refined_query)

                context = find_match(refined_query)
                # st.code(context)

                response = conversation.predict(input=f"Context:\n {context} \n\n Query:\n{query}")

            st.session_state.requests.append(query)
            st.session_state.responses.append(response)


    with response_container:
        if st.session_state['responses']:
            for i in range(len(st.session_state['responses'])):
                message(st.session_state['responses'][i],key=str(i))
                if i < len(st.session_state['requests']):
                    message(st.session_state["requests"][i], is_user=True,key=str(i)+ '_user')

except openai.error.AuthenticationError:
    errormessage = "Please provide a valid OpenAI API Key."
    st.write(errormessage)