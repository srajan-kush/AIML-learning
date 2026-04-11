exp = [2340,2500,3100,5000,2980]

# total = 0
# for item in exp:
#     total = total + item
# print(total)


# for i in range(1,11):
#     print(i*i,end=" ")

total = 0
for i in range(len(exp)):
    print('Month:',(i + 1),'Expense:',exp[i])
    total = total + exp[i]

print(total)