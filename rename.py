import os
from tqdm import tqdm
from time import sleep
 
print("\n\n")
j = 0
path = "./mnist_png/training"
for m in tqdm(range(10)):
    for i in range(10000):
        try :
            if (os.path.isfile(f"{path}/{m}/{i}.png")):
                os.rename(f"{path}/{m}/{i}.png" , f"{path}/{m}/{j}.png")
                # print(f"file {i}.png changed to {j}.png")
                j += 1

        except : 
            print("error!")
    
        finally:
                i += 1
    sleep(.1)
    j = 0
    m +=1

print("\n\nALL NAMES CHANGED AND SORTED!!!\n")