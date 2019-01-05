#env 3.7 
from pathlib import Path 
from PIL import Image 

def find_text_in_image(imgPath): 

    image = Image.open(imgPath)     
    newImage = Image.new("RGB", image.size) 
    imagePixels = newImage.load() 

    for i in range(image.size[0]):         
        for j in range(image.size[1]):  
			#in case pixel is not char ?
            if bin(image.getpixel((i, j))[0])[-1] == '0': 
                imagePixels[i, j] = (0, 0, 0)                 
            else: 
                imagePixels[i, j] = (255,255,255) 
                
    newImgPath=str(Path(imgPath).parent.absolute()) 
    newImage.save(newImgPath+'/result.png')
	
print('Here we GO!')	
find_text_in_image('C:\\Projects\\GIT\\Python\\find-text\\image.png')
