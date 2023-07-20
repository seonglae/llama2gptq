from re import split

import torch
import streamlit as st
from streamlit_chat import message

from llama2gptq.qa import qa, load_model, load_db
from llama2gptq.ingest import extract_ref

DEVICE = 'cuda'
TITLE = 'LLaMa2 GPTQ'
HUG = 'https://em-content.zobj.net/source/microsoft-teams/363/hugging-face_1f917.png'
ANGRY = 'https://em-content.zobj.net/source/microsoft-teams/363/pouting-face_1f621.png'


st.set_page_config(page_title=TITLE)
st.header(TITLE)
st.markdown('''
### Ask anythig to [Texonom](https://texonom.com).
Question for recently learned
''', unsafe_allow_html=True)


@st.cache_resource
def load_transformer():
  return (load_model(DEVICE), load_db(DEVICE))


transformer, db = load_transformer()

styl = """
<style>
    .stTextInput {
      position: fixed;
      bottom: 3rem;
      z-index: 1;
    }
    .StatusWidget-enter-done{
      position: fixed;
      left: 50%;
      top: 50%;
      transform: translate(-50%, -50%);
    }
    .StatusWidget-enter-done button{
      display: none;
    }
</style>
"""

BTN_STYLE = """
color: #aaa;
padding-right: 0.5rem;
"""


st.markdown(styl, unsafe_allow_html=True)

if 'generated' not in st.session_state:
  st.session_state['generated'] = []

if 'past' not in st.session_state:
  st.session_state['past'] = []

if 'answers' not in st.session_state:
  st.session_state['answers'] = []


def query(query):
  st.session_state.past.append(query)
  history = []
  for i, _ in enumerate(st.session_state['generated']):
    history.append([st.session_state['past'][i],
                   st.session_state["generated"][i]])

  answer, refs = qa(query, DEVICE, db, transformer, history)

  # Append references
  st.session_state.generated.append(answer)

  # Generate HTML
  answer += '<hr style="border: 1px solid #424242;"> References: '
  for ref in refs:
    ref_info = extract_ref(ref)
    btn = f"<a href='{ref_info['link']}' style='{BTN_STYLE}'>{ref_info['title']}</a>"
    answer += btn

  st.session_state.answers.append(answer)
  return answer


def get_text():
  input_text = st.text_input("You: ", key="input")
  return input_text


user_input = get_text()


if user_input:
  query(user_input)


if st.session_state['generated']:
  for i, _ in enumerate(st.session_state['generated']):
    message(st.session_state['past'][i], is_user=True,
            key=str(i) + '_user', logo=HUG)
    message(st.session_state["answers"][i],
            key=str(i), logo=ANGRY, allow_html=True)
