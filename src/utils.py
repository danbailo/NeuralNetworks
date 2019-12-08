import numpy as np
from PIL import Image
from PIL import ImageDraw 
import glob

def get_data(directory, class_):
	X = []
	Y = []
	for image_data in sorted(glob.glob(directory)):
		img = np.asarray(Image.open(image_data))
		img = np.reshape(img, -1)
		X.append(img)
		if class_ == "cat":
			Y.append(1)
		elif class_ == "noncat":
			Y.append(0)
	return X, Y

def save_img(img_np, file_name, predict):
	img = Image.fromarray(img_np[:,:,:])

	draw = ImageDraw.Draw(img)
	fillcolor = "black"
	shadowcolor = "white"
	x, y = 1, 1

	text = "{:.3f}".format(predict)

	# thin border
	draw.text((x-1, y), text, fill=shadowcolor)
	draw.text((x+1, y), text, fill=shadowcolor)
	draw.text((x, y-1), text, fill=shadowcolor)
	draw.text((x, y+1), text, fill=shadowcolor)

	# thicker border
	draw.text((x-1, y-1), text, fill=shadowcolor)
	draw.text((x+1, y-1), text, fill=shadowcolor)
	draw.text((x-1, y+1), text, fill=shadowcolor)
	draw.text((x+1, y+1), text, fill=shadowcolor)

	draw.text((x, y), text, fill=fillcolor)

	img.save(file_name)