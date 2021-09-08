import numpy as np

y=[1,-2,3,4,5,-5,7,8,-9,-10]
direction_all=[]
if len(y) >= 10:
					# sum  of xi - mean(xi-1)
    try:
        
        for index,i in enumerate(y):
            #for x in y[:index+1]:
            prev_mean= np.mean(y[:index+1])
            direc= i - prev_mean
            direction_all.append(direc)
            #print(direction_all)
        if all([x >= 0 for x in direction_all]):
            direction = 1
        elif all([x <= 0 for x in direction_all]):    
            direction = -1
        else:
            direction = 0
        print(direction)
            #if index < 1:
               # value= i
            #else:
            #    value= i - 
                
    except:
        pass
    #else:
    #pass
