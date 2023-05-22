import streamlit as st
import torch
from chat_model import *
import user_chat_model

context = torch.zeros((1, 1), dtype=torch.long, device=device)
chatting_length = 2000
next = False

st.title("Own Chat 🤖")

st.markdown('''This model learns form the chat messages given and detects the style and tone of the conversation and replyes back as person 2. 
You can feed your own chat in this model in the given formate:

```
person 1: Hi person 2
person 2: Hello person 1
so on....
```

Click 'Start to train' for pre installed chat (lang: Hinglish)''')

start = st.button("Start to train model (It will take some time)")

if start:
    st.markdown('#### Model training has been started, here you can see the stats ####')
    main()

    st.markdown(":green[AI has been trained Click on 'Genrate Chat' to se the AI genrated chat]")
    st.write("#### AI genrated chats")

    genrate = st.button('Genrate Chat')
    
    if genrate:
        st.write(f'''
        ```{decode(m.generate(context, max_new_tokens=chatting_length)[0].tolist())}```
        ''')

    st.write(' ')
    st.write("Above chat might not make a lot of sense but try to feed your own chats and see the result.")

user_chat_model.main()










