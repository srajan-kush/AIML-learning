# f = open("Python-Basics/funny.txt",'w')
# f.write("I Love Javascript")
# f.close()


# apending open with a
# f = open("Python-Basics/funny.txt",'a')
# f.write("\nI Love PHP")
# f.close()



f = open("Python-Basics/funny.txt",'r')
# print(f.read())
# f.close()
f_out = open("Python-Basics/funny_wc.txt",'w')

for line in f:
    tokens = line.split(' ')
    f_out.write("wordcount: "+str(len(tokens)) + " "+ line)
    print(len(tokens))

f.close()
f_out.close()
