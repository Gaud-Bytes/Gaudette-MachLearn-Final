#Seperates Data from classification Y is the class of Heart Disease, X is the Vital Sign Data

def read(filename):
    f = open(filename, 'r')
    next(f)
    data = []
    classification = []
    for line in f:
        tokens = line.split(',')
        error = False
        
        for i in tokens:
            if(i == '?'):
               error = True
               break
            
        if(error):
            continue
            
        for i in range(0, len(tokens)): 
            if(i != 13):
                data.append(tokens[i])
                if(i != 12):
                    data.append(',')
                    
        data.append('\n')
        


        classification.append(tokens[13])


    fwrite("Heart_Disease_X.csv", data)
    fwrite("Heart_Disease_y.csv", classification)
    f.close()


def fwrite(filename, dataList):
    f = open(filename, 'w')
    for i in range(0, len(dataList)):
        f.write(str(dataList[i]))

    f.close()


read("Heart_Disease.csv")
