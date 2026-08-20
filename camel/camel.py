camelname=input("camelCase: ")
snakename=""
for i in camelname:
    if(i.isupper()):
        i="_"+i.lower()
    snakename+=i
print("snake_case: ",snakename)