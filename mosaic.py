
#!pip install tensorflow
import numpy as np
import tensorflow as tf
import PIL.Image as Image
import glob
import matplotlib.pyplot as plt

SCALE_FACTOR = 10 # The scale factor to scale up the original image
TILESIZE = 64 #The size of the tile. 

def main(source_img_dir = "source.jpg", to_img_dir = "my_result.jpg"):
    source_img = (Image.open(source_img_dir))
    new_width = int(source_img.width * SCALE_FACTOR)
    new_height = int(source_img.height * SCALE_FACTOR)
    source_img = source_img.resize((new_width, new_height))
    source_img = np.array(source_img)
    original_width, original_height = source_img.shape[0:2]
    tiles = load_tile_images()
    tiles_rgb = cal_rgb(tiles)
    tiles_number = ((original_width//TILESIZE), (original_height//TILESIZE))
    
    img_rgb_grid, grid_locs = to_rgb_grid(np.array(source_img),tiles_number)
    #print(img_rgb_grid)
    closest_tiles = closest_tile(img_rgb_grid,tiles_rgb).numpy()
    #print(closest_tiles)
    mosaic = Image.new('RGB', (tiles_number[0]*TILESIZE,tiles_number[1]*TILESIZE))
    for (i,index) in enumerate(closest_tiles):#please pardon my chaotic var naming.
        #print(index)
        #print(tiles[index].numpy())
        #print(grid_locs[index])
        mosaic.paste(Image.fromarray(tiles[index].numpy().astype(np.uint8)),(grid_locs[i][0],grid_locs[i][2]))
    mosaic = mosaic.transpose(Image.ROTATE_270)
    mosaic.save(to_img_dir)
    plt.imshow(mosaic)

def load_tile_images():
    tile_names = sorted(glob.glob(f"tiles/*.png")) # This is your dir of your tile image.
    tiles = []
    def load_image(image_dir):
        return (Image.open(image_dir).resize((TILESIZE,TILESIZE)))
    for filedir in tile_names:
        tiles.append(load_image(filedir))
    return tf.stack(tiles)

def to_rgb_grid(img: np.array,
            tiles_number: tuple, 
            tilesize = TILESIZE):
    grid = []
    grid_locs = []
    for i in range(tiles_number[0]):
        for j in range (tiles_number[1]):
            loc = (i*tilesize,(i+1)*tilesize,j*tilesize,(j+1)*tilesize)
            grid.append(img[loc[0]:loc[1],loc[2]:loc[3],:])
            grid_locs.append(loc)
    #print(loc)
    grid = tf.stack(grid)
    return cal_rgb(grid), grid_locs

def cal_rgb(tiles: tf.Tensor):
    return tf.cast(tf.reduce_mean(tiles, axis=(1, 2)),dtype = tf.float32)

# This is the hardest part because I have to rewrite the algorism to maximize the efficiency.
def closest_tile(x:tf.Tensor,y:tf.Tensor):
    # broadcasting
    x_expand = tf.expand_dims(x, 1)  
    y_expand = tf.expand_dims(y, 0)

    #print(f"{x_expand.dtype}")

    # Calculate the distance
    distances = tf.reduce_sum(tf.square(x_expand - y_expand), axis=-1)

    # Find the index of the closest point in y for each point in x
    closest_indices = tf.argmin(distances, axis=1)

    return closest_indices



main()