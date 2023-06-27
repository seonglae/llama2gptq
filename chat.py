from re import split

import torch
import streamlit as st
from streamlit_chat import message

from src.qa import qa


DEVICE = 'cuda'
TITLE = 'Angry Face'


st.set_page_config(page_title=TITLE)
st.header(TITLE)
st.markdown('''
### Ask anythig to [Texonom](https://texonom.com).
Question for recently learned
''', unsafe_allow_html=True)

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

  _, answer_refs, answer, output_refs = qa(
      query, DEVICE, None, None, None, history)

  # Append references
  refs = answer_refs + output_refs
  ref_set = {split(r"\\|/", ref.metadata["source"])[-1] for ref in refs}
  st.session_state.generated.append(answer)

  for ref in ref_set:
    slug = split(r" |.md", ref)[-2]
    answer += f"\n> https://texonom.com/{slug}"

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
    message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
    message(st.session_state["answers"][i], key=str(i), seed=13)
