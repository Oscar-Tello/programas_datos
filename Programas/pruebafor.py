tex = 'hola mundo #ufhjr,)isjudsas!$%&/()'
diccionarioSimb = '!$=&()*,-.“”?¿¡|°¬:{}[]¨<>~^♀♂!'
diccionarioSimb2 = {'!','$','=','&','(',')','*',',','-','.','“','”','?','¿','¡','|','°','¬',':','{','}','[',']',
                   '¨','<','>','~','^','♀','♂','!'}

string = "Hey! What's up bro?"

new_string = ' '.join(char for char in string if char.isalnum())
print(new_string)

for simbolo in diccionarioSimb2:
    tex = tex.replace(simbolo, '')

print(tex)

for element in tex:
    if element !='a':
        print(element)


        #for elemento in row.split():
         #   for unidad in elemento:



def limpieza(csv):
    for row in csv:
        for simbolo in diccionarioSimb:
            print(simbolo)
            row.replace(simbolo)
    return row