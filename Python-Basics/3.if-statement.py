# num = input("Enter a number: ")
# num = int(num)

# if num%2 == 0:
#     print("even")
# else:
#     print("odd")


indian = ["samosa","daal","naan"]
chinese = ["egg role","pot sticker","fried rice"]
italian = ["pizza","pasta","risotto"]


dish = input("Enter a dish: ")

if dish in indian:
    print("Indian")
elif dish in italian:
    print("italian")
elif dish in chinese:
    print("chinese")
else:
    print("None of these")
