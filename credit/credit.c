#include <cs50.h>
#include <stdio.h>

int main(void)
{
    //initialization
    long input = 0;
    int temp = 0 ;
    int indicator = 10;
    int count = 1;
    int sum1 = 0;
    int sum2 = 0;
    int total = 0;
    int flag = 1;
    int judge = 0;
    //get first digit number
    input = get_long("Number:");
    temp = input % 10;
    input = (input - temp) / 10;
    sum2 += temp;
    // printf("%li\n",input);
    //get each digit number
    do
    {
        count ++;
        temp = input % 10;
        //judge the card type
        if (input < 100 && flag)
        {
            judge = input;
            flag = 0;
        }
        //move to next digit
        input = (input - temp) / 10;
        //add different sum
        if (count % 2 == 0)
        {
            if ((temp * 2) >= 10)
            {
                sum1 += (temp * 2) % 10 + 1;
            }
            else
            {
                sum1 += temp * 2;
            }

        }
        else
        {
            sum2 += temp;
        }
    }
    while (input);
    // printf("sum1: %i\n",sum1);
    // printf("sum2: %i\n",sum2);
    total = sum1 + sum2;
    // printf("count: %i\n",count);
    if (count == 13 || count == 15 || count == 16)
    {
        if (total % 10 == 0)
        {
            if (count == 13 && temp == 4)
            {
                printf("VISA\n");
            }
            else if (count == 15 && (judge == 34 || judge == 37))
            {
                printf("AMEX\n");
            }
            else if (count == 16 && temp == 4)
            {
                printf("VISA\n");
            }
            else if (count == 16 && (judge == 51 || judge == 52 || judge == 53 || judge == 54 || judge == 55))
            {
                printf("MASTERCARD\n");
            }
            else
            {
                printf("INVALID\n");
            }
        }
        else
        {
            printf("INVALID\n");
        }
    }
    else
    {
        printf("INVALID\n");
    }

}