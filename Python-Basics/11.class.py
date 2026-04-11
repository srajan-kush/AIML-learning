
class Human:
    def __init__(self,name,occupation):
        self.name = name
        self.occupation = occupation

    def do_work(self):
        if self.occupation == "student":
            print(self.name, "is Studying")
        elif self.occupation == "Actor":
            print(self.name, "is Acting")

    def speak(self):
        print(self.name,"says How are you?")


# tom = Human("tom","Actor")
# tom.do_work()
# tom.speak()

# maria = Human("maria khan", "student")
# maria.do_work()
# maria.speak()




# inheritence

class Vehicle:
    def general_usage(self):
        print("general use: transportation")

class Car(Vehicle):
    def __init__(self):
        print("I am Car")
        self.wheel = 4
        self.has_roof = True

    def specific_usages(self):
        print("specific use: compute to work")

class Motorcycle(Vehicle):
    def __init__(self):
        print("I am Motorcycle")
        self.wheel = 2
        self.has_roof = True

    def specific_usages(self):
        print("specific use: compute to work for 2 pasangers")



c = Car()
c.general_usage()
c.specific_usages()

m = Motorcycle()
m.general_usage()
m.specific_usages()


