import sys
import inflect
p = inflect.engine()
def ask_input():
    initial = []
    while(1):
        try:
            userInput = input()
            initial.append(userInput)
        except EOFError:
            return initial
def main():
    userInput = p.join((ask_input()))
    print("Adieu, adieu, to", userInput)

main()
