from PIL import Image

def accurate_transparent(img_path, out_path):
    img = Image.open(img_path).convert('L') # Convert to grayscale
    # Since it's white on dark background, the grayscale value is practically the alpha mask
    # Let's map the dark background to 0 alpha, and pure white to 255 alpha.
    
    # Get min/max
    extrema = img.getextrema()
    min_val, max_val = extrema
    
    # Create the alpha channel
    alpha = img.point(lambda p: max(0, min(255, int((p - min_val) * 255 / (max_val - min_val)))))
    
    # The actual image should be pure white everywhere, with just alpha varying
    out = Image.new('RGBA', img.size, (255, 255, 255, 255))
    out.putalpha(alpha)
    
    out.save(out_path, 'PNG')

accurate_transparent(r'd:\University\Sem 5 - Winter\CS719 - Data Science Project\Project\HAIDS\static\assets\images\haids-logo.png', r'd:\University\Sem 5 - Winter\CS719 - Data Science Project\Project\HAIDS\static\assets\images\haids-logo.png')
print('Transparent image generated.')
