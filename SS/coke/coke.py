due=50
while True:
    print("Amount due: ",due)
    amount=int(input("Insert coin: "))
    if amount==25 or amount==10 or amount==5:
        due-=amount
        if due<=0:
            print("Change owed: ",-due)
            break



