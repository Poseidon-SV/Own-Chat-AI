data = open('WhatsApp_Chat_with_Yash_IT.txt', encoding="utf8")
fil_data = open('filltered_data.txt', 'w',encoding="utf8")

print(type(data))
for d in data:
    if 'Missed voice call' in d or 'Missed video call' in d or 'Missed group video call' in d:
        d = None
    elif 'This message was deleted' in d:
        d = None
    elif 'Person 2:' in d:
        d = d.replace(d[:d.index('Person 2:')],'')
    elif 'Person 1:' in d:
        d = d.replace(d[:d.index('Person 1:')],'')

    if d: 
        fil_data.write(d)

fil_data.close()