from PIL import Image

def remove_background(img_path, out_path):
    img = Image.open(img_path).convert('RGBA')
    datas = img.getdata()
    
    newData = []
    for item in datas:
        # Calculate brightness
        brightness = (item[0] + item[1] + item[2]) / 3.0
        
        # The background in the image is dark gray/black. 
        # The logo and text is white.
        # If brightness is low (e.g. < 100), it's background -> transparent
        if brightness < 150:
            newData.append((255, 255, 255, 0))
        else:
            # For the white parts, we can use the brightness to determine alpha for smooth edges
            # Map brightness from 150-255 to alpha 0-255
            alpha = int(((brightness - 150) / (255 - 150)) * 255)
            # Ensure max 255
            alpha = max(0, min(255, alpha))
            newData.append((255, 255, 255, alpha))
            
    img.putdata(newData)
    img.save(out_path, 'PNG')

remove_background(r'd:\University\Sem 5 - Winter\CS719 - Data Science Project\Project\HAIDS\static\assets\images\haids-logo.png', r'd:\University\Sem 5 - Winter\CS719 - Data Science Project\Project\HAIDS\static\assets\images\haids-logo.png')
print('Perfected transparency script completed.')
