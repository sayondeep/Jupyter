def border(img):
    x=1
    cropped_array = img[x:img.shape[0]-x, x:img.shape[1]-x, :]
    return np.pad(cropped_array, ((x, x), (x, x), (0, 0)))
#%%
from operator import add
from functools import reduce

def split4(image):
    half_split = np.array_split(image, 2)
    print(half_split[0].shape)
    res = map(lambda x: np.array_split(x, 2, axis=1), half_split)
    print(res)
    return reduce(add, res)
#%%
def calculate_mean(img):
    return np.mean(img, axis=(0, 1))
#%%
def concatenate4(north_west, north_east, south_west, south_east):
    top = np.concatenate((north_west, north_east), axis=1)
    bottom = np.concatenate((south_west, south_east), axis=1)
    return np.concatenate((top, bottom), axis=0)
#%%
def split(array, nrows, ncols):
    """Split a matrix into sub-matrices."""

    r, h = array.shape
    return (array.reshape(h//nrows, nrows, -1, ncols)
            .swapaxes(1, 2)
            .reshape(-1, nrows, ncols))
#%%
def checkEqual(myList):
    first=myList[0]
    return all((x==first).all() for x in myList)

class QuadTree:

    def insert(self, img,matrix,level = 0):
        self.level = level
        #self.mean = calculate_mean(img).astype(int)
        #self.resolution = (img.shape[0], img.shape[1])
        self.img=img
        self.final = True

        if not checkEqual(matrix):
            split_img = split4(img)

            self.final = False

            n_w, n_e, s_w, s_e =  split(matrix, matrix.shape[0]//2, matrix.shape[1]//2)

            self.north_west = QuadTree().insert(split_img[0],n_w, level + 1)
            self.north_east = QuadTree().insert(split_img[1],n_e, level + 1)
            self.south_west = QuadTree().insert(split_img[2],s_w, level + 1)
            self.south_east = QuadTree().insert(split_img[3],s_e, level + 1)

        return self

    def get_image(self, level):
        if(self.final or self.level == level):
            #return np.tile(self.mean, (self.resolution[0], self.resolution[1], 1))
            # plt.imshow(self.img)
            # plt.show()
            return border(self.img)

        return concatenate4(
            self.north_west.get_image(level),
            self.north_east.get_image(level),
            self.south_west.get_image(level),
            self.south_east.get_image(level))